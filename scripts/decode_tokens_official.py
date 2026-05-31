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
    parser.add_argument("--tokens", required=True, help="Token .npz produced by songgeneration_mlx.cli.")
    parser.add_argument("--output", required=True, help="Output FLAC/WAV path.")
    parser.add_argument("--device", default=os.environ.get("SONGGEN_DEVICE", "mps"))
    parser.add_argument("--gen-type", default="mixed", choices=["mixed", "vocal", "bgm"])
    return parser.parse_args()


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
    ckpt_path = Path(args.ckpt_path).expanduser().resolve() if args.ckpt_path else repo / "songgeneration_v2_medium"
    if not (ckpt_path / "config.yaml").exists():
        raise FileNotFoundError(
            f"missing official checkpoint config: {ckpt_path / 'config.yaml'}; "
            "download an official SongGeneration checkpoint and pass --ckpt-path"
        )
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "codeclm" / "tokenizer"))
    sys.path.insert(0, str(repo / "codeclm" / "tokenizer" / "Flow1dVAE"))

    OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx], replace=True)
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)), replace=True)

    from codeclm.models import builders
    from codeclm.models.codeclm import CodecLM

    device = torch.device(args.device)
    cfg = OmegaConf.load(ckpt_path / "config.yaml")
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
