from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .model import generate_tokens, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SongGeneration audio tokens with MLX.")
    parser.add_argument("--model", required=True, help="Converted SongGeneration MLX model directory.")
    parser.add_argument("--lyrics", required=True, help="Lyrics text.")
    parser.add_argument("--description", required=True, help="Song style/type description.")
    parser.add_argument("--duration", type=float, default=2.0, help="Generation duration in seconds.")
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", required=True, help="Output .npz token file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, metadata = load_model(args.model)
    tokens = generate_tokens(
        model,
        tokenizer,
        lyrics=args.lyrics,
        description=args.description,
        duration=args.duration,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        tokens=np.array(tokens),
        metadata=json.dumps(
            {
                "source": metadata.get("source"),
                "duration": args.duration,
                "top_k": args.top_k,
                "temperature": args.temperature,
            },
            ensure_ascii=False,
        ),
    )
    mx.eval(tokens)
    print(f"wrote {output} shape={tokens.shape}")


if __name__ == "__main__":
    main()
