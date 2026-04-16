import argparse
import copy
import time

import numpy as np
import torch

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.profiling import merge_perf_runs, run_profiled_inference


class ModuleTimer:
    def __init__(self, module: torch.nn.Module, device: str = "cuda"):
        self.module = module
        self.device = torch.device(device)
        self.calls = []
        self.original_forward = module.forward

    def __enter__(self) -> "ModuleTimer":
        def wrapped_forward(*args, **kwargs):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = self.original_forward(*args, **kwargs)
                end.record()
                torch.cuda.synchronize(self.device)
                self.calls.append(float(start.elapsed_time(end)))
                return out

            t0 = time.perf_counter()
            out = self.original_forward(*args, **kwargs)
            self.calls.append((time.perf_counter() - t0) * 1000.0)
            return out

        self.module.forward = wrapped_forward
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.module.forward = self.original_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Alpamayo-R1 inference latency.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/workspace/.cache/modelscope/hub/models/nv-community/Alpamayo-R1-10B",
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default="100ae358-f548-49b8-af4d-c0afdbcfe9ed",
    )
    parser.add_argument(
        "--ncore-manifest-path",
        type=str,
        default="/workspace/dataset/pai_100ae358-f548-49b8-af4d-c0afdbcfe9ed.json",
    )
    parser.add_argument("--ncore-root", type=str, default="/workspace/dataset")
    parser.add_argument(
        "--extract-cache-dir",
        type=str,
        default="/tmp/alpamayo_ncore_extract",
    )
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--profile-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--num-traj-sets", type=int, default=1)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--diffusion-inference-step", type=int, default=None)
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    args = parse_args()
    dtype = get_dtype(args.dtype)

    print(f"Loading dataset for clip_id: {args.clip_id}...")
    data = load_physical_aiavdataset(
        clip_id=args.clip_id,
        t0_us=args.t0_us,
        ncore_manifest_path=args.ncore_manifest_path,
        ncore_root=args.ncore_root,
        extract_cache_dir=args.extract_cache_dir,
    )
    print("Dataset loaded.")

    print(f"Loading model from {args.model_path}...")
    model = AlpamayoR1.from_pretrained(args.model_path, dtype=dtype).to(args.device)
    print(model.vlm)
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    print("Model loaded.")

    sample_kwargs = {
        "top_p": args.top_p,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "num_traj_samples": args.num_traj_samples,
        "num_traj_sets": args.num_traj_sets,
        "max_generation_length": args.max_generation_length,
        "return_extra": True,
    }
    if args.diffusion_inference_step is not None:
        sample_kwargs["diffusion_kwargs"] = {"inference_step": args.diffusion_inference_step}

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    print(f"Running {args.warmup_runs} warmup runs...")
    for warmup_idx in range(args.warmup_runs):
        with ModuleTimer(model.vlm.model.visual, device=args.device) as vision_timer:
            _, _, extra = run_profiled_inference(
                model=model,
                data=copy.deepcopy(data),
                message_builder=helper.create_message,
                processor=processor,
                device=args.device,
                autocast_dtype=dtype,
                sample_kwargs=sample_kwargs,
            )
        if vision_timer.calls:
            extra.setdefault("perf_raw", {}).setdefault("vision_encode_ms", []).extend(vision_timer.calls)
            vision_mean = sum(vision_timer.calls) / len(vision_timer.calls)
            extra.setdefault("perf_summary", {})["vision_encode_ms"] = {
                "count": float(len(vision_timer.calls)),
                "mean_ms": vision_mean,
                "std_ms": 0.0,
                "min_ms": min(vision_timer.calls),
                "p50_ms": vision_mean,
                "p90_ms": vision_mean,
                "p99_ms": vision_mean,
                "max_ms": max(vision_timer.calls),
            }
        e2e_mean = extra["perf_summary"].get("e2e_ms", {}).get("mean_ms")
        if e2e_mean is not None:
            if vision_timer.calls:
                print(
                    f"  warmup {warmup_idx + 1}/{args.warmup_runs}: "
                    f"e2e={e2e_mean:.3f} ms, vision={vision_timer.calls[0]:.3f} ms"
                )
            else:
                print(f"  warmup {warmup_idx + 1}/{args.warmup_runs}: e2e={e2e_mean:.3f} ms")

    print(f"Running {args.profile_runs} profiled runs...")
    profiled_extras = []
    last_pred_xyz = None
    last_extra = None
    for run_idx in range(args.profile_runs):
        with ModuleTimer(model.vlm.model.visual, device=args.device) as vision_timer:
            pred_xyz, _, extra = run_profiled_inference(
                model=model,
                data=copy.deepcopy(data),
                message_builder=helper.create_message,
                processor=processor,
                device=args.device,
                autocast_dtype=dtype,
                sample_kwargs=sample_kwargs,
            )
        if vision_timer.calls:
            extra.setdefault("perf_raw", {}).setdefault("vision_encode_ms", []).extend(vision_timer.calls)
        profiled_extras.append(extra)
        last_pred_xyz = pred_xyz
        last_extra = extra
        vision_msg = ""
        if vision_timer.calls:
            vision_ms = vision_timer.calls[0]
            vlm_ms = extra["perf_summary"]["vlm_generate_ms"]["mean_ms"]
            vision_ratio = 100.0 * vision_ms / vlm_ms if vlm_ms > 0 else 0.0
            vision_msg = f", vision={vision_ms:.3f} ms ({vision_ratio:.1f}% of vlm)"
        print(
            f"  run {run_idx + 1}/{args.profile_runs}: "
            f"e2e={extra['perf_summary']['e2e_ms']['mean_ms']:.3f} ms, "
            f"vlm={extra['perf_summary']['vlm_generate_ms']['mean_ms']:.3f} ms, "
            f"diff={extra['perf_summary']['diffusion_total_ms']['mean_ms']:.3f} ms"
            f"{vision_msg}"
        )

    merged = merge_perf_runs(profiled_extras)

    print("\nLatency report (aggregated across profiled runs):")
    print(merged["perf_report"])
    if "vision_encode_ms" in merged["perf_summary"]:
        vision_ms = merged["perf_summary"]["vision_encode_ms"]["mean_ms"]
        vlm_ms = merged["perf_summary"]["vlm_generate_ms"]["mean_ms"]
        e2e_ms = merged["perf_summary"]["e2e_ms"]["mean_ms"]
        print(
            "\nVision encode summary: "
            f"mean={vision_ms:.3f} ms, "
            f"share_of_vlm={100.0 * vision_ms / vlm_ms:.1f}%, "
            f"share_of_e2e={100.0 * vision_ms / e2e_ms:.1f}%"
        )

    if last_extra is not None and "cot" in last_extra:
        print("\nChain-of-Causation (last run):")
        print(last_extra["cot"][0])

    if last_pred_xyz is not None:
        gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
        pred_xy = last_pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
        min_ade = diff.min()
        print(f"\nminADE (last run): {min_ade} meters")


if __name__ == "__main__":
    main()
