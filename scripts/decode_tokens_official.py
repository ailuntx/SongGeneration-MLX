#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[1] / "third_party" / "SongGeneration"
    parser = argparse.ArgumentParser(description="Decode MLX SongGeneration tokens with the official PyTorch decoder.")
    parser.add_argument("--repo", default=str(default_repo), help="Official SongGeneration checkout.")
    parser.add_argument("--ckpt-path", help="Official checkpoint directory, e.g. songgeneration_v2_medium.")
    parser.add_argument("--mlx-model", help="MLX checkpoint directory containing config.official.yaml.")
    parser.add_argument("--tokens", required=True, help="Token .npz produced by songgeneration_mlx.cli.")
    parser.add_argument("--output", required=True, help="Output FLAC/WAV path.")
    parser.add_argument("--device", default=os.environ.get("SONGGEN_DEVICE", "mps"))
    parser.add_argument("--gen-type", default="mixed", choices=["mixed", "vocal", "bgm"])
    return parser.parse_args()


def resolve_existing(repo: Path, value: str) -> str:
    if not value.startswith("./"):
        return value
    rel = value[2:]
    for base in (repo, repo / "runtime"):
        candidate = base / rel
        if candidate.exists():
            return str(candidate.resolve())
    return str((repo / rel).resolve())


def resolve_tokenizer_checkpoint(repo: Path, value: str) -> str:
    if "_" not in value:
        return resolve_existing(repo, value)
    prefix, path = value.split("_", 1)
    return f"{prefix}_{resolve_existing(repo, path)}"


def save_audio(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    wav = wav.detach().cpu().float()
    audio = wav.numpy()
    if audio.ndim == 2:
        audio = audio.T
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    if args.ckpt_path:
        config_path = Path(args.ckpt_path).expanduser().resolve() / "config.yaml"
    elif args.mlx_model:
        config_path = Path(args.mlx_model).expanduser().resolve() / "config.official.yaml"
    else:
        config_path = repo / "songgeneration_v2_medium" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"missing official config: {config_path}; "
            "pass --mlx-model for an MLX checkpoint or --ckpt-path for an official checkpoint"
        )
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "runtime"))
    sys.path.insert(0, str(repo / "runtime" / "third_party"))
    sys.path.insert(0, str(repo / "codeclm" / "tokenizer"))
    sys.path.insert(0, str(repo / "codeclm" / "tokenizer" / "Flow1dVAE"))

    OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx], replace=True)
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)), replace=True)

    from codeclm.models import builders
    from codeclm.models.codeclm import CodecLM

    device = torch.device(args.device)
    cfg = OmegaConf.load(config_path)
    cfg.audio_tokenizer_checkpoint_sep = resolve_tokenizer_checkpoint(repo, cfg.audio_tokenizer_checkpoint_sep)
    cfg.vae_config = resolve_existing(repo, cfg.vae_config)
    cfg.vae_model = resolve_existing(repo, cfg.vae_model)
    cfg.runtime_device = str(device)
    cfg.mode = "inference"

    token_data = np.load(args.tokens)
    tokens = torch.from_numpy(token_data["tokens"]).long().to(device)
    tokenizer = builders.get_audio_tokenizer_model_cpu(cfg.audio_tokenizer_checkpoint_sep, cfg)
    tokenizer.model.device = device
    tokenizer.model.vae = tokenizer.model.vae.to(device)
    tokenizer.model.model.device = device
    tokenizer.model.model = tokenizer.model.model.to(device)
    tokenizer = tokenizer.eval()

    model = CodecLM(name="mlx_bridge", lm=None, audiotokenizer=None, max_duration=cfg.max_dur, seperate_tokenizer=tokenizer)
    ctx = torch.autocast(device_type=device.type, dtype=torch.float16) if device.type in ("cuda", "mps") else nullcontext()
    with torch.no_grad(), ctx:
        wav = model.generate_audio(tokens, chunked=True, gen_type=args.gen_type)
    save_audio(Path(args.output), wav[0], int(cfg.sample_rate))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
