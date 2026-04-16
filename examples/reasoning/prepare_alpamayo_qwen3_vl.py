import argparse
import sys
from pathlib import Path

from transformers import AutoModelForVision2Seq, AutoProcessor

WORKSPACE_ROOT = Path("/workspace")
ALPAMAYO_SRC = WORKSPACE_ROOT / "alpamayo" / "src"
if str(ALPAMAYO_SRC) not in sys.path:
    sys.path.insert(0, str(ALPAMAYO_SRC))

from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.base_model import SPECIAL_TOKENS, TRAJ_TOKEN


def export_from_alpamayo_checkpoint(
    alpamayo_ckpt: str,
    output_dir: str,
    dtype: str,
) -> None:
    model = AlpamayoR1.from_pretrained(alpamayo_ckpt, dtype=dtype)
    vlm = model.vlm
    tokenizer = model.tokenizer
    processor = helper.get_processor(tokenizer)

    new_vocab_size = len(tokenizer)
    print(f"Original vocab size: {vlm.config.vocab_size}, new vocab size after loading Alpamayo checkpoint: {new_vocab_size}")
    vlm.resize_token_embeddings(new_vocab_size)
    if hasattr(vlm.config, "text_config"):
        vlm.config.text_config.vocab_size = new_vocab_size
    vlm.config.vocab_size = new_vocab_size

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vlm.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    print(f"Exported Alpamayo VLM to: {output_path}")
    print(f"Tokenizer vocab size: {new_vocab_size}")


def export_from_base_model(
    base_model: str,
    output_dir: str,
    traj_vocab_size: int,
    add_special_tokens: bool,
) -> None:
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    tokenizer = processor.tokenizer

    discrete_tokens = [f"<i{i}>" for i in range(traj_vocab_size)]
    tokenizer.add_tokens(discrete_tokens)

    if add_special_tokens:
        tokenizer.add_tokens(list(SPECIAL_TOKENS.values()), special_tokens=True)
    else:
        tokenizer.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)

    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    new_vocab_size = len(tokenizer)
    model.resize_token_embeddings(new_vocab_size)
    if hasattr(model.config, "text_config"):
        model.config.text_config.vocab_size = new_vocab_size
    model.config.vocab_size = new_vocab_size

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)

    print(f"Saved Alpamayo-ready Qwen3-VL model to: {output_path}")
    print(f"Tokenizer vocab size: {new_vocab_size}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a standalone VLM checkpoint for RLinf. "
        "Preferred path: export directly from a full Hugging Face Alpamayo checkpoint."
    )
    parser.add_argument(
        "--alpamayo-ckpt",
        default="/workspace/.cache/modelscope/hub/models/nv-community/Alpamayo-R1-10B",
        help="Full Hugging Face Alpamayo checkpoint path. Preferred when available.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Fallback base Qwen3-VL model path or HF id.",
    )
    parser.add_argument("--output-dir", help="Export directory",default="/workspace/Alpamayo-R1-10B-vlm")
    parser.add_argument("--dtype", default="bfloat16", help="dtype passed to AlpamayoR1.from_pretrained")
    parser.add_argument("--traj-vocab-size", type=int, default=4000)
    parser.add_argument("--add-special-tokens", action="store_true")
    args = parser.parse_args()

    if args.alpamayo_ckpt:
        export_from_alpamayo_checkpoint(
            alpamayo_ckpt=args.alpamayo_ckpt,
            output_dir=args.output_dir,
            dtype=args.dtype,
        )
        return

    if args.base_model:
        export_from_base_model(
            base_model=args.base_model,
            output_dir=args.output_dir,
            traj_vocab_size=args.traj_vocab_size,
            add_special_tokens=args.add_special_tokens,
        )
        return

    raise ValueError("You must provide either --alpamayo-ckpt or --base-model.")


if __name__ == "__main__":
    main()
