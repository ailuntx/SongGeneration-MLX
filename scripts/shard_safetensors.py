#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)([KMG]i?B?|B)?$", re.IGNORECASE)


def parse_size(value: str) -> int:
    match = SIZE_RE.match(value.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"invalid size: {value}")
    number = float(match.group(1))
    unit = (match.group(2) or "B").lower()
    multipliers = {
        "b": 1,
        "kb": 1000,
        "kib": 1024,
        "mb": 1000**2,
        "mib": 1024**2,
        "gb": 1000**3,
        "gib": 1024**3,
    }
    return int(number * multipliers[unit])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shard SongGeneration MLX safetensors weights.")
    parser.add_argument("model_dir", help="Converted MLX model directory containing safetensors weights.")
    parser.add_argument("--max-shard-size", type=parse_size, default=parse_size("256MiB"))
    parser.add_argument("--remove-source", action="store_true", help="Delete model.safetensors after shards are written.")
    return parser.parse_args()


def tensor_nbytes(tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def source_weight_files(model_dir: Path) -> list[Path]:
    source = model_dir / "model.safetensors"
    if source.exists():
        return [source]
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index["weight_map"].values()))
        return [model_dir / name for name in names]
    shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(f"no safetensors weights found in {model_dir}")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    source_files = source_weight_files(model_dir)

    shard_dir = model_dir / ".shard_tmp"
    shard_dir.mkdir(exist_ok=True)
    for old in shard_dir.glob("*.safetensors"):
        old.unlink()

    index: dict[str, str] = {}
    total_size = 0
    pending = {}
    pending_size = 0
    shard_id = 1
    shard_paths: list[Path] = []

    def flush() -> None:
        nonlocal pending, pending_size, shard_id
        if not pending:
            return
        path = shard_dir / f"model-{shard_id:05d}.safetensors"
        save_file(pending, path)
        for key in pending:
            index[key] = path.name
        shard_paths.append(path)
        shard_id += 1
        pending = {}
        pending_size = 0

    for source in source_files:
        with safe_open(source, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                size = tensor_nbytes(tensor)
                total_size += size
                if pending and pending_size + size > args.max_shard_size:
                    flush()
                pending[key] = tensor
                pending_size += size
    flush()

    total_shards = len(shard_paths)
    width = max(5, int(math.log10(total_shards)) + 1)
    final_names: list[str] = []
    for i, path in enumerate(shard_paths, start=1):
        final_name = f"model-{i:0{width}d}-of-{total_shards:0{width}d}.safetensors"
        final_path = model_dir / final_name
        if final_path.exists():
            final_path.unlink()
        path.replace(final_path)
        final_names.append(final_name)
        for key, name in list(index.items()):
            if name == path.name:
                index[key] = final_name

    shard_dir.rmdir()
    index_path = model_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": index}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config_path = model_dir / "config.json"
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    metadata["weight_files"] = final_names
    config_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path = model_dir / "mlx_manifest.json"
    if manifest_path.exists():
        manifest_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.remove_source:
        for source in source_files:
            if source.exists() and source.name not in final_names:
                source.unlink()
    print(f"wrote {total_shards} shards to {model_dir}")


if __name__ == "__main__":
    main()
