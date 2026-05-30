#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx

from songgeneration_mlx.model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize a converted SongGeneration MLX LM directory.")
    parser.add_argument("--source", required=True, help="Converted unquantized MLX directory.")
    parser.add_argument("--output", required=True, help="Output quantized MLX directory.")
    parser.add_argument("--bits", type=int, required=True, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=64)
    return parser.parse_args()


def copy_metadata(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ["config.official.yaml", "vocab.yaml", "README.md"]:
        src = source / name
        if src.exists():
            shutil.copy2(src, output / name)
    qwen_src = source / "qwen2_tokenizer"
    qwen_dst = output / "qwen2_tokenizer"
    if qwen_dst.exists():
        shutil.rmtree(qwen_dst)
    shutil.copytree(qwen_src, qwen_dst)


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    copy_metadata(source, output)

    model, _, metadata = load_model(source)
    model.apply_quantization(bits=args.bits, group_size=args.group_size)
    mx.eval(model.parameters())
    model.save_weights(str(output / "model.safetensors"))

    metadata = dict(metadata)
    metadata["precision"] = f"{args.bits}bit"
    metadata["quantization"] = {
        "bits": args.bits,
        "group_size": args.group_size,
        "mode": "affine",
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "mlx_manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
