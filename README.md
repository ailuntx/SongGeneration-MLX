# SongGeneration-MLX

Apple MLX runtime and conversion tools for Tencent SongGeneration.

This repository targets the heavy autoregressive `audiolm` token generator in
SongGeneration-v2. The audio decoder is still bridged through the official
PyTorch Flow1dVAE / separate tokenizer path.

The official SongGeneration source tree is vendored under
`third_party/SongGeneration` so token decoding and upstream reference code live
in the same repository. Model checkpoints and runtime assets are still external
and must be downloaded separately.

## Status

- `songgeneration_v2_medium` and `songgeneration_v2_large` official PyTorch/MPS baselines have been inspected locally.
- MLX conversion covers SongGeneration-v2 medium and large language model weights.
- MLX runtime generates discrete song tokens.
- Published checkpoints use sharded safetensors to keep individual upload/download
  objects small and resumable.
- Full FLAC decoding still uses the official PyTorch decoder as a bridge.
- The recent-token repetition penalty from the official sampler is required. Without it,
  long generations collapse into repeated tokens and decode close to silence.
- Public upstream sources checked on 2026-05-31 did not provide downloadable
  SongGeneration-v2-fast weights. Fast is intentionally not published here until
  a real upstream fast checkpoint is available.

## Current Validation

Tested locally on Apple Silicon:

| Path | Result |
|---|---|
| MLX LM token generation, 2s | 300 pattern steps, 38.85s wall time, output shape `(1, 3, 50)` |
| MLX LM token generation, 12s | 550 pattern steps, about 1 minute wall time, output shape `(1, 3, 300)` |
| Official PyTorch/MPS decoder bridge, 12s | 73.27s wall time |
| Final 12s FLAC | 48 kHz, stereo, 12.000s, FLAC/PCM16, RMS about `0.163` |

Compared with the earlier PyTorch/MPS baseline, the LM token phase dropped from
roughly 3:52-4:08 for a 12s sample to about 1 minute in this first MLX runtime.
The decoder is not MLX yet.

## Install

```bash
git clone https://github.com/ailuntx/SongGeneration-MLX.git
cd SongGeneration-MLX
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -U huggingface_hub hf_transfer
```

## Download MLX Weights

Pick one checkpoint:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download mlx-community/SongGeneration-v2-medium-4bit --local-dir ./models/SongGeneration-v2-medium-4bit
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download mlx-community/SongGeneration-v2-medium-8bit --local-dir ./models/SongGeneration-v2-medium-8bit
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download mlx-community/SongGeneration-v2-medium-bfloat16 --local-dir ./models/SongGeneration-v2-medium-bfloat16
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download mlx-community/SongGeneration-v2-medium-fp32 --local-dir ./models/SongGeneration-v2-medium-fp32

HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download mlx-community/SongGeneration-v2-large-4bit --local-dir ./models/SongGeneration-v2-large-4bit
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download mlx-community/SongGeneration-v2-large-8bit --local-dir ./models/SongGeneration-v2-large-8bit
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download mlx-community/SongGeneration-v2-large-bfloat16 --local-dir ./models/SongGeneration-v2-large-bfloat16
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download mlx-community/SongGeneration-v2-large-fp32 --local-dir ./models/SongGeneration-v2-large-fp32
```

Published checkpoints:

| Variant | 4-bit | 8-bit | bfloat16 | fp32 |
|---|---|---|---|---|
| v2-medium | yes | yes | yes | yes |
| v2-large | yes | yes | yes | yes |
| v2-fast | pending upstream weights | pending upstream weights | pending upstream weights | pending upstream weights |

## Generate Tokens

```bash
.venv/bin/python -m songgeneration_mlx.cli \
  --model ./models/SongGeneration-v2-medium-4bit \
  --lyrics "[verse] Hello from MLX. [chorus] Sing it again." \
  --description "Pop, female vocal, bright production, [Musicality-medium]." \
  --duration 2 \
  --top-k 50 \
  --temperature 0.9 \
  --output ./tokens_2s.npz
```

## Convert Locally

```bash
python scripts/convert_lm.py \
  --source /path/to/SongGeneration/songgeneration_v2_medium \
  --repo /path/to/SongGeneration \
  --output ./models/SongGeneration-v2-medium-bfloat16 \
  --variant v2-medium \
  --dtype bfloat16

PYTHONPATH=. python scripts/quantize_lm.py \
  --source ./models/SongGeneration-v2-medium-bfloat16 \
  --output ./models/SongGeneration-v2-medium-4bit \
  --bits 4

python scripts/shard_safetensors.py \
  ./models/SongGeneration-v2-medium-4bit \
  --max-shard-size 64MiB \
  --remove-source
```

## Prepare Official Decoder Runtime

The MLX runtime generates discrete song tokens. FLAC decoding still uses the
official PyTorch Flow1dVAE / separate-tokenizer runtime.

Install the official decoder dependencies in a PyTorch environment:

```bash
python -m venv .venv-decoder
.venv-decoder/bin/pip install -U pip
.venv-decoder/bin/pip install \
  -r third_party/SongGeneration/requirements.txt \
  -r third_party/SongGeneration/requirements_nodeps.txt \
  soundfile
```

Download the official runtime assets into the vendored source tree:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download tencent/SongGeneration \
  --include "runtime/*" \
  --local-dir ./third_party/SongGeneration
```

If the runtime is already available elsewhere, a symlink is enough:

```bash
ln -sfn /path/to/SongGeneration/runtime ./third_party/SongGeneration/runtime
```

## Decode Tokens With Official Bridge

The bridge uses the vendored official source in `third_party/SongGeneration` by
default. When `--mlx-model` is provided, it reads `config.official.yaml` from the
MLX checkpoint, so the original SongGeneration `model.pt` is not needed for
decoding.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 SONGGEN_DEVICE=mps \
.venv-decoder/bin/python scripts/decode_tokens_official.py \
  --mlx-model ./models/SongGeneration-v2-medium-4bit \
  --tokens ./tokens_2s.npz \
  --output ./output_2s.flac \
  --device mps
```

## Limitations

- The `audiolm` token generator is MLX.
- The final audio decoder is still the official PyTorch bridge.
- The bridge needs official `runtime/` assets, but it does not need the
  original SongGeneration LM `model.pt` when `--mlx-model` is used.
- Fast checkpoints are not published because upstream fast weights were not
  available in the public release checked on 2026-05-31.

## License

License follows the upstream SongGeneration release. Check the official model
card and repository for the authoritative model license.

## Citation

```bibtex
@misc{songgeneration-mlx,
  title  = {SongGeneration-MLX: Apple MLX port of SongGeneration},
  author = {ailuntx},
  year   = {2026},
  url    = {https://github.com/ailuntx/SongGeneration-MLX},
}

@article{lei2025levo,
  title   = {LeVo: High-Quality Song Generation with Multi-Preference Alignment},
  author  = {Lei, Shun and Xu, Yaoxun and Lin, Zhiwei and Zhang, Huaicheng and Tan, Wei and Chen, Hangting and Yu, Jianwei and Zhang, Yixuan and Yang, Chenyu and Zhu, Haina and Wang, Shuai and Wu, Zhiyong and Yu, Dong},
  journal = {arXiv preprint arXiv:2506.07520},
  year    = {2025},
}
```
