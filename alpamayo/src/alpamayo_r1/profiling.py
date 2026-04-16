import copy
import statistics
import time
from collections import defaultdict
from contextlib import ContextDecorator
from typing import Any

import einops
import numpy as np
import torch
from transformers import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

from alpamayo_r1.diffusion.flow_matching import FlowMatching
from alpamayo_r1.models.alpamayo_r1 import ExpertLogitsProcessor
from alpamayo_r1.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)


def _sync_if_needed(device: torch.device | str | None) -> None:
    if device is None:
        return
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer(ContextDecorator):
    def __init__(self, device: torch.device | str | None = None):
        self.device = torch.device(device) if device is not None else None
        self.elapsed_ms = 0.0
        self._start_event = None
        self._end_event = None
        self._t0 = 0.0

    def __enter__(self) -> "Timer":
        if self.device is not None and self.device.type == "cuda":
            _sync_if_needed(self.device)
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if self.device is not None and self.device.type == "cuda":
            self._end_event.record()
            _sync_if_needed(self.device)
            self.elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0


def _append_latency(perf: dict[str, list[float]], key: str, value_ms: float) -> None:
    perf[key].append(float(value_ms))


def _summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    if len(values) == 1:
        value = float(values[0])
        return {
            "count": 1.0,
            "mean_ms": value,
            "std_ms": 0.0,
            "min_ms": value,
            "p50_ms": value,
            "p90_ms": value,
            "p99_ms": value,
            "max_ms": value,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "min_ms": float(arr.min()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
    }


def summarize_perf(perf: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    summary = {}
    for key, values in perf.items():
        stats = _summarize(values)
        if stats is not None:
            summary[key] = stats
    return summary


def format_perf_report(summary: dict[str, dict[str, float]]) -> str:
    ordered_keys = [
        "e2e_ms",
        "message_build_ms",
        "processor_apply_chat_template_ms",
        "to_device_ms",
        "traj_fuse_ms",
        "vlm_generate_ms",
        "vlm_postprocess_ms",
        "diffusion_total_ms",
        "diffusion_step_ms",
        "diffusion_action_in_proj_ms",
        "diffusion_expert_ms",
        "diffusion_action_out_proj_ms",
        "action_to_traj_ms",
        "return_extra_ms",
        "vlm_generated_tokens",
        "vlm_ms_per_generated_token",
    ]
    keys = [key for key in ordered_keys if key in summary]
    keys.extend(key for key in summary if key not in keys)

    lines = []
    header = (
        f"{'metric':36} {'count':>7} {'mean':>10} {'p50':>10} "
        f"{'p90':>10} {'p99':>10} {'max':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for key in keys:
        stats = summary[key]
        lines.append(
            f"{key:36}"
            f" {int(stats['count']):7d}"
            f" {stats['mean_ms']:10.3f}"
            f" {stats['p50_ms']:10.3f}"
            f" {stats['p90_ms']:10.3f}"
            f" {stats['p99_ms']:10.3f}"
            f" {stats['max_ms']:10.3f}"
        )
    return "\n".join(lines)


def profile_flow_matching_sample(
    diffusion: FlowMatching,
    batch_size: int,
    step_fn: Any,
    device: torch.device,
    return_all_steps: bool = False,
    inference_step: int | None = None,
    int_method: str | None = None,
) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], dict[str, list[float]]]:
    if (int_method or diffusion.int_method) != "euler":
        raise ValueError("Only euler integration is supported in profiling helper.")

    perf = defaultdict(list)
    inference_step = inference_step or diffusion.num_inference_steps
    n_dim = len(diffusion.x_dims)

    with Timer(device) as total_timer:
        x = torch.randn(batch_size, *diffusion.x_dims, device=device)
        time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
        if return_all_steps:
            all_steps = [x]

        for i in range(inference_step):
            with Timer(device) as step_timer:
                dt = time_steps[i + 1] - time_steps[i]
                dt = dt.view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
                t_start = time_steps[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
                v = step_fn(x=x, t=t_start)
                x = x + dt * v
                if return_all_steps:
                    all_steps.append(x)
            _append_latency(perf, "diffusion_step_ms", step_timer.elapsed_ms)
    _append_latency(perf, "diffusion_total_ms", total_timer.elapsed_ms)

    if return_all_steps:
        return (torch.stack(all_steps, dim=1), time_steps), perf
    return x, perf


@torch.no_grad()
def sample_trajectories_from_data_with_vlm_rollout_profiled(
    model: Any,
    data: dict[str, Any],
    top_p: float = 0.98,
    top_k: int | None = None,
    temperature: float = 0.6,
    num_traj_samples: int = 6,
    num_traj_sets: int = 1,
    diffusion_kwargs: dict[str, Any] | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    del args

    perf = defaultdict(list)
    n_samples_total = num_traj_samples * num_traj_sets
    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    bsz, n_traj_group, _, _ = ego_history_xyz.shape
    assert n_traj_group == 1, "Only one trajectory group is supported for inference."

    tokenized_data = dict(data["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")
    traj_data_vlm = {
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }

    device = input_ids.device

    with Timer(device) as fuse_timer:
        input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)
    _append_latency(perf, "traj_fuse_ms", fuse_timer.elapsed_ms)

    max_generation_length = kwargs.get(
        "max_generation_length", model.config.tokens_per_future_traj
    )
    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = num_traj_samples
    generation_config.max_new_tokens = max_generation_length
    generation_config.output_logits = True
    generation_config.return_dict_in_generate = True
    generation_config.top_k = top_k
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
    logits_processor = LogitsProcessorList(
        [
            ExpertLogitsProcessor(
                traj_token_offset=model.config.traj_token_start_idx,
                traj_vocab_size=model.config.traj_vocab_size,
            )
        ]
    )

    with Timer(device) as vlm_timer:
        vlm_outputs = model.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )
    _append_latency(perf, "vlm_generate_ms", vlm_timer.elapsed_ms)
    vlm_outputs.rope_deltas = model.vlm.model.rope_deltas

    generated_tokens = vlm_outputs.sequences.shape[1] - input_ids.shape[1]
    _append_latency(perf, "vlm_generated_tokens", float(generated_tokens))
    if generated_tokens > 0:
        _append_latency(perf, "vlm_ms_per_generated_token", vlm_timer.elapsed_ms / generated_tokens)

    with Timer(device) as post_timer:
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=model.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        b_star = vlm_outputs.sequences.shape[0]
        traj_future_start_mask = vlm_outputs.sequences == eos_token_id
        has_traj_future_start = traj_future_start_mask.any(dim=1)
        traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
        last_token_positions = torch.full(
            (b_star,), vlm_outputs.sequences.shape[1] - 1, device=device
        )
        valid_token_pos_id = torch.where(
            has_traj_future_start, traj_future_start_positions, last_token_positions
        )
        offset = valid_token_pos_id + 1

        n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
        delta = vlm_outputs.rope_deltas + offset[:, None]
        position_ids += delta.to(position_ids.device)

        attention_mask = torch.zeros(
            (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
            dtype=torch.float32,
            device=device,
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
                attention_mask.dtype
            ).min
    _append_latency(perf, "vlm_postprocess_ms", post_timer.elapsed_ms)

    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b_star_local = x.shape[0]
        with Timer(device) as action_in_proj_timer:
            future_token_embeds = model.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(
                    b_star_local, n_diffusion_tokens, -1
                )
        _append_latency(perf, "diffusion_action_in_proj_ms", action_in_proj_timer.elapsed_ms)

        with Timer(device) as expert_timer:
            expert_out_base = model.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
        _append_latency(perf, "diffusion_expert_ms", expert_timer.elapsed_ms)

        with Timer(device) as action_out_proj_timer:
            pred = model.action_out_proj(last_hidden).view(
                -1, *model.action_space.get_action_space_dims()
            )
        _append_latency(perf, "diffusion_action_out_proj_ms", action_out_proj_timer.elapsed_ms)
        return pred

    total_batch = bsz * n_samples_total
    if diffusion_kwargs is None:
        diffusion_kwargs = {}

    if isinstance(model.diffusion, FlowMatching):
        sampled_action, diffusion_perf = profile_flow_matching_sample(
            diffusion=model.diffusion,
            batch_size=total_batch,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
            inference_step=diffusion_kwargs.get("inference_step"),
            int_method=diffusion_kwargs.get("int_method"),
        )
        for key, values in diffusion_perf.items():
            perf[key].extend(values)
    else:
        with Timer(device) as diffusion_timer:
            sampled_action = model.diffusion.sample(
                batch_size=total_batch,
                step_fn=step_fn,
                device=device,
                return_all_steps=False,
                **diffusion_kwargs,
            )
        _append_latency(perf, "diffusion_total_ms", diffusion_timer.elapsed_ms)

    hist_xyz_rep = einops.repeat(
        ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
    )
    hist_rot_rep = einops.repeat(
        ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
    )

    with Timer(device) as action_to_traj_timer:
        pred_xyz, pred_rot = model.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )
    _append_latency(perf, "action_to_traj_ms", action_to_traj_timer.elapsed_ms)

    pred_xyz = einops.rearrange(
        pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
    )
    pred_rot = einops.rearrange(
        pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
    )

    extra = {"perf_raw": dict(perf), "perf_summary": summarize_perf(dict(perf))}
    if kwargs.get("return_extra", False):
        with Timer(device) as return_extra_timer:
            text_extra = extract_text_tokens(model.tokenizer, vlm_outputs.sequences)
            for text_tokens in text_extra.keys():
                text_extra[text_tokens] = np.array(text_extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
        _append_latency(perf, "return_extra_ms", return_extra_timer.elapsed_ms)
        extra.update(text_extra)
        extra["perf_raw"] = dict(perf)
        extra["perf_summary"] = summarize_perf(dict(perf))

    return pred_xyz, pred_rot, extra


def run_profiled_inference(
    model: Any,
    data: dict[str, Any],
    message_builder: Any,
    processor: Any,
    device: str,
    autocast_dtype: torch.dtype,
    sample_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    perf = defaultdict(list)

    with Timer() as message_timer:
        messages = message_builder(data["image_frames"].flatten(0, 1))
    _append_latency(perf, "message_build_ms", message_timer.elapsed_ms)

    with Timer() as processor_timer:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
    _append_latency(perf, "processor_apply_chat_template_ms", processor_timer.elapsed_ms)
    # print(inputs.keys())
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }

    from alpamayo_r1.helper import to_device

    with Timer(torch.device(device)) as to_device_timer:
        model_inputs = to_device(model_inputs, device)
    _append_latency(perf, "to_device_ms", to_device_timer.elapsed_ms)

    with Timer(torch.device(device)) as e2e_timer:
        with torch.autocast(device_type=torch.device(device).type, dtype=autocast_dtype):
            pred_xyz, pred_rot, extra = sample_trajectories_from_data_with_vlm_rollout_profiled(
                model=model,
                data=model_inputs,
                **sample_kwargs,
            )
    _append_latency(perf, "e2e_ms", e2e_timer.elapsed_ms)

    combined_perf = defaultdict(list)
    for key, values in perf.items():
        combined_perf[key].extend(values)
    for key, values in extra.get("perf_raw", {}).items():
        combined_perf[key].extend(values)

    extra["perf_raw"] = dict(combined_perf)
    extra["perf_summary"] = summarize_perf(dict(combined_perf))
    extra["perf_report"] = format_perf_report(extra["perf_summary"])
    return pred_xyz, pred_rot, extra


def merge_perf_runs(extras: list[dict[str, Any]]) -> dict[str, Any]:
    merged = defaultdict(list)
    for extra in extras:
        for key, values in extra.get("perf_raw", {}).items():
            merged[key].extend(values)
    summary = summarize_perf(dict(merged))
    return {
        "perf_raw": dict(merged),
        "perf_summary": summary,
        "perf_report": format_perf_report(summary),
    }


def summarize_run_scalar(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"mean={statistics.mean(values):.3f}, std={statistics.pstdev(values):.3f}"
