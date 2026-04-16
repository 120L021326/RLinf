import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
import einops

from transformers import AutoProcessor, AutoTokenizer

from rlinf.data.datasets.item import DatasetItem
from rlinf.data.datasets.vlm import VLMBaseDataset, VLMDatasetRegistry

from rlinf.data.datasets import load_physical_aiavdataset
from rlinf.data.datasets.delta_tokenizer import DeltaTrajectoryTokenizer


MIN_PIXELS = 163840
MAX_PIXELS = 196608
BASE_PROCESSOR_NAME = "/workspace/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct/"
TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}

def get_processor(
    tokenizer: AutoTokenizer,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> AutoProcessor:
    """Get the processor for the Qwen3-VL-2B-Instruct model."""
    processor_kwargs = {
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }

    processor = AutoProcessor.from_pretrained(BASE_PROCESSOR_NAME, **processor_kwargs)
    processor.tokenizer = tokenizer
    return processor

def create_message(frames: torch.Tensor):
    """Construct the message using images and cot."""
    if frames.ndim != 4:
        raise ValueError(f"{frames.ndim=}, expected 4 (N, C, H, W)")

    # NOTE: we expand the padding tokens to match training, so we can directly apply native processor from VLM.
    num_traj_token = 48
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": frame} for frame in frames]
            + [
                {
                    "type": "text",
                    "text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "<|cot_start|>",
                }
            ],
        },
    ]

def tokenize_history_trajectory(
    tokenizer: Any, traj_data: dict[str, Any], start_idx: int = 0
) -> torch.Tensor:
    """Tokenize the history trajectory with prefix shape of (B, n_traj, ...).

    Args:
        tokenizer: Trajectory tokenizer with encode method
        traj_data: dict containing "ego_history_xyz" and "ego_history_rot"
        start_idx: start of token index of the history trajectory tokens

    Returns:
        torch.Tensor: [B, n_traj * tokens_per_history_traj]
    """
    assert "ego_history_xyz" in traj_data
    assert traj_data["ego_history_xyz"].ndim == 4, "ego_history_xyz must be 4D of [B, n_traj, T, 3]"

    B = traj_data["ego_history_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)

    hist_idx = (
        tokenizer.encode(
            hist_xyz=hist_xyz[:, :1],
            hist_rot=hist_rot[:, :1],
            fut_xyz=hist_xyz,  # note hist_xyz is passed to fut_xyz as it's encoding history.
            fut_rot=hist_rot,
        )
        + start_idx
    )  # [B*n_traj, tokens_per_history_traj]
    hist_idx = einops.rearrange(hist_idx, "(b n_traj) n -> b (n_traj n)", b=B)

    return hist_idx

def _frame_to_png_bytes(frame_chw: torch.Tensor) -> bytes:
    frame_hwc = frame_chw.permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(frame_hwc)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
@VLMDatasetRegistry.register("alpamayo_coc")
class AlpamayoCoCDataset(VLMBaseDataset):
    """Minimal Alpamayo RL dataset.

    Expected per-record keys:
    - `clip_id` (required)
    - `t0_us` / `ncore_manifest_path` / `ncore_root` / `extract_cache_dir` (optional)
    - `answer` / `coc` / `target` (optional, currently unused by reward)

    Data loading follows Alpamayo's test_inference path:
    `load_physical_aiavdataset -> create_message -> processor.apply_chat_template`.
    """

    def __init__(self, data_paths, config, tokenizer) -> None:
        super().__init__(data_paths=data_paths, config=config, tokenizer=tokenizer)
        self.min_pixels = int(config.data.get("min_pixels", MIN_PIXELS))
        self.max_pixels = int(config.data.get("max_pixels", MAX_PIXELS))
        if self.min_pixels <= 0 or self.max_pixels <= 0:
            raise ValueError(
                f"config.data min/max pixels must be positive, got {self.min_pixels=} {self.max_pixels=}"
            )
        if self.min_pixels > self.max_pixels:
            raise ValueError(
                f"config.data.min_pixels ({self.min_pixels}) must be <= max_pixels ({self.max_pixels})"
            )
        self.processor = get_processor(
            tokenizer,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self.hist_traj_tokenizer = DeltaTrajectoryTokenizer()
        self.hist_traj_token_start_idx = int(config.data.get("hist_traj_token_start_idx", 154669))
        self.hist_placeholder_id = self.tokenizer.convert_tokens_to_ids(TRAJ_TOKEN["history"])
        self.default_t0_us = 5_100_000
        self.default_num_frames = int(config.data.get("num_frames", 4))
        camera_features = config.data.get("camera_features", None)
        self.default_camera_features = (
            list(camera_features) if camera_features is not None else None
        )
        self.default_ncore_manifest_path = config.data.get("ncore_manifest_path", None)
        self.default_ncore_root = config.data.get("ncore_root", None)
        self.default_extract_cache_dir = config.data.get("extract_cache_dir", None)

        if self.default_num_frames <= 0:
            raise ValueError(
                f"config.data.num_frames must be positive, got {self.default_num_frames}"
            )

        if self.hist_traj_token_start_idx < 0 or self.hist_placeholder_id < 0:
            raise ValueError(
                "Tokenizer is missing Alpamayo trajectory tokens. "
                "Run the Qwen3-VL preparation script first and point config paths to the exported model."
            )

    def _load_sample_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        clip_id = raw.get("clip_id")
        if not clip_id:
            raise ValueError("Each Alpamayo sample must contain `clip_id`.")

        return load_physical_aiavdataset.load_physical_aiavdataset(
            clip_id=clip_id,
            t0_us=int(raw.get("t0_us", self.default_t0_us)),
            camera_features=raw.get("camera_features", self.default_camera_features),
            num_frames=int(raw.get("num_frames", self.default_num_frames)),
            ncore_manifest_path=raw.get("ncore_manifest_path", self.default_ncore_manifest_path),
            ncore_root=raw.get("ncore_root", self.default_ncore_root),
            extract_cache_dir=raw.get("extract_cache_dir", self.default_extract_cache_dir),
        )

    def _build_inputs(
        self, data: dict[str, Any]
    ) -> tuple[torch.Tensor, str, list[bytes], dict[str, Any]]:
        frames = data["image_frames"].flatten(0, 1)
        messages = create_message(frames)
        image_data = [_frame_to_png_bytes(frame) for frame in frames]
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = tokenized.pop("input_ids").squeeze(0).to(dtype=torch.long)
        tokenized.pop("attention_mask", None)
        multi_modal_inputs = {k: v for k, v in tokenized.items()}
        return input_ids, rendered, image_data, multi_modal_inputs

    def _process_raw_record(self, raw: dict[str, Any], idx: int) -> DatasetItem:
        data = self._load_sample_data(raw)
        prompt_ids, rendered, image_data, multi_modal_inputs = self._build_inputs(data)

        traj_data = {
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        hist_idx = tokenize_history_trajectory(
            self.hist_traj_tokenizer,
            traj_data,
            self.hist_traj_token_start_idx,
        ).squeeze(0)

        mask = prompt_ids == self.hist_placeholder_id
        if int(mask.sum().item()) != int(hist_idx.numel()):
            raise ValueError(
                "Trajectory placeholder count does not match encoded history token count: "
                f"{int(mask.sum().item())} vs {int(hist_idx.numel())}"
            )
        prompt_ids = prompt_ids.masked_scatter(mask, hist_idx.to(prompt_ids.device))

        answer = raw.get("answer") or raw.get("coc") or raw.get("target") or ""
        meta = {
            "clip_id": raw["clip_id"],
            "t0_us": data.get("t0_us"),
            "hist_token_count": int(hist_idx.numel()),
        }
        return DatasetItem(
            prompt=prompt_ids,
            length=int(prompt_ids.numel()),
            answer=answer,
            idx=idx,
            solution=raw.get("solution"),
            image_data=image_data,
            prompt_text=rendered,
            meta=meta,
            multi_modal_inputs=multi_modal_inputs,
        )
