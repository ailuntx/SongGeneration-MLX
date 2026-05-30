# SongGeneration-MLX

Apple MLX runtime and conversion tools for Tencent SongGeneration.

This repository currently targets the heavy autoregressive `audiolm` token generator in
SongGeneration-v2. The audio decoder is still bridged through the official PyTorch
Flow1dVAE / separate tokenizer path.

## Status

- `songgeneration_v2_medium` official PyTorch/MPS baseline has been tested locally.
- MLX conversion covers the SongGeneration-v2 medium language model weights.
- MLX runtime generates discrete song tokens.
- Full FLAC decoding still uses the official PyTorch decoder as a bridge.
- The recent-token repetition penalty from the official sampler is required. Without it,
  long generations collapse into repeated tokens and decode close to silence.

## Current Validation

Tested locally on Apple Silicon:

| Path | Result |
|---|---|
| MLX LM token generation, 2s | 300 pattern steps, 38.85s wall time, output shape `(1, 3, 50)` |
| MLX LM token generation, 12s | 550 pattern steps, 59.95s wall time, output shape `(1, 3, 300)` |
| Official PyTorch/MPS decoder bridge, 12s | 73.27s wall time |
| Final 12s FLAC | 48 kHz, stereo, 12.000s, FLAC/PCM16, RMS about `0.163` |

Reference audio for listening:

```text
/Volumes/usb_main/home/index_mlx/SongGeneration-MLX-12s-zh-pop.flac
```

Compared with the earlier PyTorch/MPS baseline, the LM token phase dropped from
roughly 3:52-4:08 for a 12s sample to about 1:00 in this first MLX runtime.
The decoder is not MLX yet.

## Convert

```bash
/Users/ailuntz/miniforge3/envs/env_songgeneration/bin/python scripts/convert_lm.py \
  --source /Volumes/usb_main/home/server_llm/experiments_audio_generation/third_party/SongGeneration/songgeneration_v2_medium \
  --repo /Volumes/usb_main/home/server_llm/experiments_audio_generation/third_party/SongGeneration \
  --output /Volumes/usb_main/home/index_mlx/models/SongGeneration-v2-medium-MLX \
  --variant v2-medium
```

## Generate Tokens

```bash
/Volumes/usb_main/home/server_llm/.venv_eval/bin/python -m songgeneration_mlx.cli \
  --model /Volumes/usb_main/home/index_mlx/models/SongGeneration-v2-medium-MLX \
  --lyrics "[verse] Hello from MLX. [chorus] Sing it again." \
  --description "Pop, female vocal, bright production, [Musicality-medium]." \
  --duration 2 \
  --output /Volumes/usb_main/home/index_mlx/runs/SongGeneration-MLX/tokens_2s.npz
```

Use `PYTHONPATH=/Volumes/usb_main/home/index_mlx/project/SongGeneration-MLX` if the package is not installed editable.

## Decode Tokens With Official Bridge

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 SONGGEN_DEVICE=mps \
/Users/ailuntz/miniforge3/envs/env_songgeneration/bin/python scripts/decode_tokens_official.py \
  --repo /Volumes/usb_main/home/server_llm/experiments_audio_generation/third_party/SongGeneration \
  --ckpt-path /Volumes/usb_main/home/server_llm/experiments_audio_generation/third_party/SongGeneration/songgeneration_v2_medium \
  --tokens /Volumes/usb_main/home/index_mlx/runs/SongGeneration-MLX/tokens_12s_zh_pop_record.npz \
  --output /Volumes/usb_main/home/index_mlx/runs/SongGeneration-MLX/mlx_tokens_12s_zh_pop_record.flac \
  --device mps
```
