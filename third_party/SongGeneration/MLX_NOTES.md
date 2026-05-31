# SongGeneration Source Snapshot

This directory is based on `tencent-ailab/SongGeneration` commit `b1b03ec`.

It was copied from the local adapted checkout at
`/Volumes/usb_main/home/server_llm/experiments_audio_generation/third_party/SongGeneration`,
not from a fresh upstream clone. That local checkout already contained the
Apple MPS-compatible decoder bridge changes used by this repository.

This directory keeps the upstream source code, small sample files, and local
Apple MPS-compatible decoder bridge changes. Large checkpoint and runtime assets
are not tracked here:

- `runtime/`
- `ckpt`
- `third_party`
- `songgeneration_v2_*`

Download `runtime/` separately when running the official PyTorch decoder. The
decoder can read `config.official.yaml` from an MLX checkpoint via
`scripts/decode_tokens_official.py --mlx-model`, so the original SongGeneration
LM `model.pt` is not required for the decode bridge.
