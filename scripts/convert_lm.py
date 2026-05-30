#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SongGeneration audiolm weights for the MLX runtime.")
    parser.add_argument("--source", required=True, help="Official SongGeneration checkpoint directory.")
    parser.add_argument("--repo", required=True, help="Official SongGeneration source code checkout.")
    parser.add_argument("--output", required=True, help="Output MLX model directory.")
    parser.add_argument("--variant", default="v2-medium")
    return parser.parse_args()


def map_key(key: str) -> str | None:
    if not key.startswith("audiolm."):
        return None
    k = key.removeprefix("audiolm.")
    replacements = {
        "emb.0.weight": "emb0.weight",
        "condition_provider.conditioners.prompt_audio.EOT_emb": "prompt_audio_eot",
        "condition_provider.conditioners.prompt_audio.layer2_EOT_emb": "prompt_audio_layer2_eot",
        "condition_provider.conditioners.prompt_audio.emb.0.weight": "prompt_audio_emb0.weight",
        "condition_provider.conditioners.prompt_audio.emb.1.weight": "prompt_audio_emb1.weight",
        "condition_provider.conditioners.prompt_audio.emb.2.weight": "prompt_audio_emb2.weight",
        "condition_provider.conditioners.description.output_proj.weight": "description_output_proj.weight",
        "condition_provider.conditioners.description.structure_emb.weight": "description_structure_emb.weight",
        "condition_provider.conditioners.type_info.output_proj.weight": "type_info_output_proj.weight",
        "layer2_emb.0.weight": "layer2_emb0.weight",
        "layer2_emb.1.weight": "layer2_emb1.weight",
        "layer2_emb.2.weight": "layer2_emb2.weight",
        "mlp.0.weight": "mlp.linear0.weight",
        "mlp.0.bias": "mlp.linear0.bias",
        "mlp.2.weight": "mlp.linear2.weight",
        "mlp.2.bias": "mlp.linear2.bias",
        "transformer.lm_head.weight": "transformer_lm_head.weight",
        "linears.0.weight": "linears0.weight",
        "linears.1.weight": "linears1.weight",
    }
    if k in replacements:
        return replacements[k]
    if k.startswith("transformer.model."):
        return "transformer." + k.removeprefix("transformer.model.")
    if k.startswith("transformer2.model."):
        return "transformer2." + k.removeprefix("transformer2.model.")
    if k.startswith("transformer2.lm_head.") or k.startswith("transformer2.model.embed_tokens."):
        return None
    return None


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    state = torch.load(source / "model.pt", map_location="cpu", weights_only=False)
    weights = {}
    skipped = []
    for key, value in state.items():
        new_key = map_key(key)
        if new_key is None:
            skipped.append(key)
            continue
        if not torch.is_tensor(value):
            continue
        weights[new_key] = value.contiguous()

    save_file(weights, output / "model.safetensors")
    shutil.copy2(source / "config.yaml", output / "config.official.yaml")
    shutil.copy2(repo / "conf" / "vocab.yaml", output / "vocab.yaml")
    qwen_src = repo / "runtime" / "third_party" / "Qwen2-7B"
    qwen_dst = output / "qwen2_tokenizer"
    if qwen_dst.exists():
        shutil.rmtree(qwen_dst)
    qwen_dst.mkdir(parents=True)
    for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json", "config.json"]:
        src = qwen_src / name
        if src.exists():
            shutil.copy2(src, qwen_dst / name)

    metadata = {
        "format": "songgeneration-mlx-audiolm",
        "source": "tencent/SongGeneration",
        "official_code": "https://github.com/tencent-ailab/songgeneration",
        "variant": args.variant,
        "runtime": {
            "hidden_size": cfg["lm"]["dim"],
            "intermediate_size": cfg["lm"]["intermediate_size"],
            "num_heads": cfg["lm"]["num_heads"],
            "num_layers": cfg["lm"]["num_layers"],
            "num_layers_sub": cfg["lm"]["num_layers_sub"],
            "code_depth": cfg["lm"]["code_depth"],
            "code_size": cfg["lm"]["code_size"],
            "prompt_len": cfg["prompt_len"],
            "frame_rate": cfg["audio_tokenizer_frame_rate"],
            "max_position_embeddings": cfg["lm"]["max_position_embeddings"],
            "max_position_embeddings_sub": cfg["lm"]["max_position_embeddings_sub"],
            "rope_theta": cfg["lm"]["rope_theta"],
            "rope_theta_sub": cfg["lm"]["rope_theta_sub"],
            "cfg_coef": cfg["classifier_free_guidance"]["inference_coef"],
        },
        "components": {
            "audiolm": "converted to MLX safetensors",
            "qwen_tokenizer": "copied for lyric/style tokenization",
            "audio_decoder": "use official PyTorch Flow1dVAE/separate tokenizer bridge for now",
        },
        "skipped_keys": skipped,
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "mlx_manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"weights: {len(weights)} tensors; skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
