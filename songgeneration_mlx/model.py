from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import yaml
from mlx.utils import tree_unflatten
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.llama import LlamaModel, ModelArgs
from transformers import Qwen2Tokenizer


@dataclass
class RuntimeConfig:
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_heads: int = 12
    num_layers: int = 28
    num_layers_sub: int = 12
    code_depth: int = 3
    code_size: int = 16384
    prompt_len: int = 10
    frame_rate: int = 25
    max_position_embeddings: int = 10000
    max_position_embeddings_sub: int = 10000
    rope_theta: float = 500000.0
    rope_theta_sub: float = 500000.0
    cfg_coef: float = 1.5
    lyric_max_len: int = 600
    type_max_len: int = 100
    pad_token_id: int = 151643
    special_token_id: int = 16385
    eos_token_id: int = 16384

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class SongGenerationMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear0 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.gelu(self.linear0(x)))


class SongGenerationLM(nn.Module):
    def __init__(self, config: RuntimeConfig, add_token_list: list[str]):
        super().__init__()
        self.config = config
        self.add_token_list = add_token_list
        d = config.hidden_size
        input_emb_dim = config.code_size + 2
        vocab = config.code_size + 1

        llama_args = ModelArgs(
            model_type="llama",
            hidden_size=d,
            num_hidden_layers=config.num_layers,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            rms_norm_eps=1e-5,
            vocab_size=vocab,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=False,
        )
        llama_args_sub = ModelArgs(
            model_type="llama",
            hidden_size=d,
            num_hidden_layers=config.num_layers_sub,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            rms_norm_eps=1e-5,
            vocab_size=vocab,
            max_position_embeddings=config.max_position_embeddings_sub,
            rope_theta=config.rope_theta_sub,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=False,
        )

        self.emb0 = nn.Embedding(input_emb_dim, d)
        self.layer2_emb0 = nn.Embedding(input_emb_dim, d)
        self.layer2_emb1 = nn.Embedding(input_emb_dim, d)
        self.layer2_emb2 = nn.Embedding(input_emb_dim, d)
        self.prompt_audio_emb0 = nn.Embedding(input_emb_dim, d)
        self.prompt_audio_emb1 = nn.Embedding(input_emb_dim, d)
        self.prompt_audio_emb2 = nn.Embedding(input_emb_dim, d)
        self.description_output_proj = nn.Embedding(151659, d)
        self.description_structure_emb = nn.Embedding(200, d)
        self.type_info_output_proj = nn.Embedding(151652, d)
        self.prompt_audio_eot = mx.zeros((1, d))
        self.prompt_audio_layer2_eot = mx.zeros((1, d))
        self.transformer = LlamaModel(llama_args)
        self.transformer_lm_head = nn.Linear(d, vocab, bias=False)
        self.transformer2 = LlamaModel(llama_args_sub)
        self.mlp = SongGenerationMLP(d)
        self.linears0 = nn.Linear(d, vocab, bias=False)
        self.linears1 = nn.Linear(d, vocab, bias=False)
        self.reset_cache()

    @property
    def code_depth(self) -> int:
        return self.config.code_depth

    def reset_cache(self) -> None:
        self.cache1 = make_prompt_cache(self.transformer)
        self.cache2 = make_prompt_cache(self.transformer2)
        self.offset = 0

    def load_weights_file(self, path: str | Path) -> None:
        weights = mx.load(str(path))
        self.update(tree_unflatten(list(weights.items())))
        mx.eval(self.parameters())

    def apply_quantization(self, bits: int, group_size: int = 64) -> None:
        nn.quantize(self, group_size=group_size, bits=bits)

    def _pad_2d(self, x: mx.array, max_len: int, pad_id: int) -> mx.array:
        width = x.shape[1]
        if width > max_len:
            return x[:, :max_len]
        if width == max_len:
            return x
        pad = mx.full((x.shape[0], max_len - width), pad_id, dtype=x.dtype)
        return mx.concatenate([x, pad], axis=1)

    def _tokenize_description(self, tokenizer: Qwen2Tokenizer, texts: list[str | None]) -> tuple[mx.array, mx.array, mx.array]:
        values = ["<|im_start|>" + x if x is not None else "<|im_start|>" for x in texts]
        inputs = tokenizer(values, return_tensors="np", padding=True)
        tokens = mx.array(inputs["input_ids"], dtype=mx.int32)
        mask = mx.array(inputs["attention_mask"], dtype=mx.int32)
        vocab = tokenizer.get_vocab()
        struct_ids = [vocab[t] for t in self.add_token_list if t.startswith("[") and t.endswith("]") and t in vocab]
        tokens_np = inputs["input_ids"]
        mask_np = inputs["attention_mask"]
        import numpy as np

        cover = np.zeros_like(tokens_np)
        for b in range(tokens_np.shape[0]):
            positions = [int(i) for i in np.where(np.isin(tokens_np[b], struct_ids))[0]]
            positions.append(int(mask_np[b].sum()))
            for i, start in enumerate(positions[:-1]):
                cover[b, start : positions[i + 1]] = int(tokens_np[b, start]) - 151645
        tokens = self._pad_2d(tokens, self.config.lyric_max_len, self.config.pad_token_id)
        mask = self._pad_2d(mask, self.config.lyric_max_len, 0)
        cover_mx = self._pad_2d(mx.array(cover, dtype=mx.int32), self.config.lyric_max_len, 0)
        emb = self.description_output_proj(tokens) + self.description_structure_emb(cover_mx)
        return emb, emb, mask

    def _tokenize_type_info(self, tokenizer: Qwen2Tokenizer, texts: list[str | None]) -> tuple[mx.array, mx.array, mx.array]:
        values = ["<|im_start|>" + x if x is not None else "<|im_start|>" for x in texts]
        inputs = tokenizer(values, return_tensors="np", padding=True)
        tokens = self._pad_2d(mx.array(inputs["input_ids"], dtype=mx.int32), self.config.type_max_len, self.config.pad_token_id)
        mask = self._pad_2d(mx.array(inputs["attention_mask"], dtype=mx.int32), self.config.type_max_len, 0)
        emb = self.type_info_output_proj(tokens)
        return emb, emb, mask

    def _prompt_audio_condition(self, audio_qt_emb: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        bsz = audio_qt_emb.shape[0]
        eos = mx.full((bsz, self.config.code_depth, 1), self.config.eos_token_id, dtype=mx.int32)
        seq = mx.concatenate([eos, audio_qt_emb.astype(mx.int32)], axis=-1)
        mask = audio_qt_emb[:, :, 0:1] == self.config.special_token_id
        mask = mx.repeat(mask, seq.shape[-1], axis=-1)
        seq = mx.where(mask, self.config.special_token_id, seq)
        max_body = self.config.prompt_len * self.config.frame_rate + 1
        seq = seq[:, :, :max_body]
        if seq.shape[-1] < max_body:
            pad = mx.full((bsz, self.config.code_depth, max_body - seq.shape[-1]), self.config.special_token_id, dtype=mx.int32)
            seq = mx.concatenate([seq, pad], axis=-1)
        emb1 = self.prompt_audio_emb0(seq[:, 0])
        eot1 = mx.repeat(mx.expand_dims(self.prompt_audio_eot, 0), bsz, axis=0)
        emb1 = mx.concatenate([eot1, emb1], axis=1)
        emb2 = self.prompt_audio_emb1(seq[:, 1]) + self.prompt_audio_emb2(seq[:, 2])
        eot2 = mx.repeat(mx.expand_dims(self.prompt_audio_layer2_eot, 0), bsz, axis=0)
        emb2 = mx.concatenate([eot2, emb2], axis=1)
        length = min(max_body + 1, emb1.shape[1])
        cond_mask = mx.zeros((bsz, emb1.shape[1]), dtype=mx.int32)
        cond_mask = mx.where(mx.arange(emb1.shape[1])[None, :] < length, 1, cond_mask)
        return emb1, emb2, cond_mask

    def prepare_condition_tensors(
        self,
        tokenizer: Qwen2Tokenizer,
        lyrics: list[str],
        descriptions: list[str],
        audio_qt_emb: mx.array | None = None,
    ) -> dict[str, tuple[mx.array, mx.array, mx.array]]:
        if audio_qt_emb is None:
            target_len = self.config.prompt_len * self.config.frame_rate
            audio_qt_emb = mx.full((1, self.config.code_depth, target_len), self.config.special_token_id, dtype=mx.int32)
        cond_texts = lyrics
        cond_descs = descriptions
        null_texts = [None for _ in lyrics]
        null_descs = [self._null_type_info(d) for d in descriptions]
        null_audio = mx.full(audio_qt_emb.shape, self.config.special_token_id, dtype=audio_qt_emb.dtype)
        doubled_audio = mx.concatenate([audio_qt_emb, null_audio], axis=0)
        return {
            "description": self._tokenize_description(tokenizer, cond_texts + null_texts),
            "prompt_audio": self._prompt_audio_condition(doubled_audio),
            "type_info": self._tokenize_type_info(tokenizer, cond_descs + null_descs),
        }

    def _null_type_info(self, text: str | None) -> str | None:
        if text is not None and "[Musicality-very-high]" in text:
            return "[Musicality-very-low], ."
        return None

    def _fuse(self, input1: mx.array, input2: mx.array, conditions: dict[str, tuple[mx.array, mx.array, mx.array]]) -> tuple[mx.array, mx.array]:
        if self.offset > 0:
            return input1, input2
        fused1, fused2 = input1, input2
        for name in reversed(["description", "prompt_audio", "type_info"]):
            cond1, cond2, _ = conditions[name]
            fused1 = mx.concatenate([cond1, fused1], axis=1)
            fused2 = mx.concatenate([cond2, fused2], axis=1)
        return fused1, fused2

    def __call__(self, sequence: mx.array, conditions: dict[str, tuple[mx.array, mx.array, mx.array]]) -> mx.array:
        bsz, depth, steps = sequence.shape
        if depth != self.config.code_depth:
            raise ValueError(f"expected code depth {self.config.code_depth}, got {depth}")
        input1 = self.emb0(sequence[:, 0])
        input2 = self.layer2_emb1(sequence[:, 1]) + self.layer2_emb2(sequence[:, 2])
        fused1, fused2 = self._fuse(input1, input2, conditions)
        hidden1 = self.transformer(mx.zeros((bsz, fused1.shape[1]), dtype=mx.int32), cache=self.cache1, input_embeddings=fused1)
        logits1 = self.transformer_lm_head(hidden1)
        fused2 = mx.concatenate([fused2, hidden1], axis=-1)
        fused2 = self.mlp(fused2)
        hidden2 = self.transformer2(mx.zeros((bsz, fused2.shape[1]), dtype=mx.int32), cache=self.cache2, input_embeddings=fused2)
        logits2 = mx.stack([self.linears0(hidden2), self.linears1(hidden2)], axis=1)
        logits = mx.concatenate([mx.expand_dims(logits1, 1), logits2], axis=1)
        self.offset += steps
        return logits[:, :, -steps:, :]

    def sample_next(
        self,
        sequence: mx.array,
        conditions: dict[str, tuple[mx.array, mx.array, mx.array]],
        cfg_coef: float | None = None,
        top_k: int = 250,
        temperature: float = 1.0,
        ignore_tokens: mx.array | None = None,
        sampled_token_pool: list[np.ndarray] | None = None,
    ) -> mx.array:
        cfg = self.config.cfg_coef if cfg_coef is None else cfg_coef
        doubled = mx.concatenate([sequence, sequence], axis=0)
        logits = self(doubled, conditions)
        cond, uncond = mx.split(logits, 2, axis=0)
        logits = uncond + (cond - uncond) * cfg
        logits = logits[:, :, -1, :]
        if sampled_token_pool:
            penalty = np.ones((self.config.code_depth, self.config.code_size + 1), dtype=np.float32)
            stacked = np.stack(sampled_token_pool[-50:], axis=-1)
            for q in range(self.config.code_depth):
                ids = np.unique(stacked[q])
                ids = ids[(ids >= 0) & (ids < self.config.code_size)]
                penalty[q, ids] = 1.1
            logits = logits / mx.array(penalty, dtype=logits.dtype)[None, :, :]
        if ignore_tokens is not None and ignore_tokens.size > 0:
            first = logits[:, 0, :]
            first = mx.scatter(first, ignore_tokens[None, :], mx.full((1, ignore_tokens.size), -mx.inf), axis=1)
            logits = mx.concatenate([first[:, None, :], logits[:, 1:, :]], axis=1)
        if temperature <= 0:
            return mx.argmax(logits, axis=-1)[:, :, None]
        first = self._sample_top_k(logits[:, 0:1, :] / temperature, top_k)
        rest = self._sample_top_k(logits[:, 1:, :] / temperature, 1)
        return mx.concatenate([first, rest], axis=1)[:, :, None]

    def _sample_top_k(self, logits: mx.array, k: int) -> mx.array:
        if k <= 0 or k >= logits.shape[-1]:
            return mx.random.categorical(logits, axis=-1).astype(mx.int32)
        values = mx.topk(logits, k=k, axis=-1)
        cutoff = values[..., -1:]
        masked = mx.where(logits < cutoff, -mx.inf, logits)
        return mx.random.categorical(masked, axis=-1).astype(mx.int32)


def load_model(model_dir: str | Path) -> tuple[SongGenerationLM, Qwen2Tokenizer, dict[str, Any]]:
    model_dir = Path(model_dir)
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = RuntimeConfig.from_dict(data["runtime"])
    with open(model_dir / "vocab.yaml", "r", encoding="utf-8") as f:
        add_token_list = yaml.safe_load(f)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_dir / "qwen2_tokenizer")
    tokenizer.add_tokens(add_token_list, special_tokens=True)
    model = SongGenerationLM(cfg, add_token_list)
    quantization = data.get("quantization")
    if quantization:
        model.apply_quantization(bits=int(quantization["bits"]), group_size=int(quantization.get("group_size", 64)))
    model.load_weights_file(model_dir / "model.safetensors")
    return model, tokenizer, data


def delayed_pattern_masks(timesteps: int, code_depth: int = 3, delays: list[int] | None = None) -> tuple[mx.array, mx.array]:
    delays = [0, 250, 250] if delays is None else delays
    layout: list[list[tuple[int, int]]] = [[]]
    for t in range(timesteps + max(delays)):
        coords = []
        for q, delay in enumerate(delays):
            tq = t - delay
            if tq >= 0:
                coords.append((tq, q))
        layout.append(coords)
    steps = len(layout)
    indexes = [[code_depth * timesteps for _ in range(steps)] for _ in range(code_depth)]
    mask = [[False for _ in range(steps)] for _ in range(code_depth)]
    for s, coords in enumerate(layout):
        for t, q in coords:
            if t < timesteps:
                indexes[q][s] = t + q * timesteps
                mask[q][s] = True
    return mx.array(indexes, dtype=mx.int32), mx.array(mask, dtype=mx.bool_)


def build_pattern_sequence(codes: mx.array, special_token: int) -> tuple[mx.array, mx.array]:
    bsz, depth, timesteps = codes.shape
    indexes, mask = delayed_pattern_masks(timesteps, depth)
    flat = mx.reshape(codes, (bsz, -1))
    flat = mx.concatenate([flat, mx.full((bsz, 1), special_token, dtype=codes.dtype)], axis=1)
    values = flat[:, mx.reshape(indexes, (-1,))]
    values = mx.reshape(values, (bsz, depth, indexes.shape[-1]))
    return values, mask


def revert_pattern_sequence(sequence: mx.array, timesteps: int, special_token: int) -> mx.array:
    bsz, depth, steps = sequence.shape
    layout: list[list[tuple[int, int]]] = [[]]
    delays = [0, 250, 250]
    for t in range(timesteps + max(delays)):
        coords = []
        for q, delay in enumerate(delays):
            tq = t - delay
            if tq >= 0:
                coords.append((tq, q))
        layout.append(coords)
    indexes = [[depth * steps for _ in range(timesteps)] for _ in range(depth)]
    for s, coords in enumerate(layout[:steps]):
        for t, q in coords:
            if t < timesteps:
                indexes[q][t] = s + q * steps
    flat = mx.reshape(sequence, (bsz, -1))
    flat = mx.concatenate([flat, mx.full((bsz, 1), special_token, dtype=sequence.dtype)], axis=1)
    values = flat[:, mx.reshape(mx.array(indexes, dtype=mx.int32), (-1,))]
    return mx.reshape(values, (bsz, depth, timesteps))


def generate_tokens(
    model: SongGenerationLM,
    tokenizer: Qwen2Tokenizer,
    lyrics: str,
    description: str,
    duration: float,
    top_k: int = 250,
    temperature: float = 1.0,
) -> mx.array:
    timesteps = int(duration * model.config.frame_rate)
    model.reset_cache()
    conditions = model.prepare_condition_tensors(tokenizer, [lyrics], [description])
    gen_codes = mx.full((1, model.config.code_depth, timesteps), -1, dtype=mx.int32)
    gen_sequence, mask = build_pattern_sequence(gen_codes, model.config.special_token_id)
    output_sequence = mx.full(gen_sequence.shape, model.config.eos_token_id, dtype=gen_sequence.dtype)
    is_end = mx.zeros((1, model.config.code_depth, 1), dtype=mx.bool_)
    sampled_token_pool: list[np.ndarray] = []
    prev_offset = 0
    start = 1
    for offset in range(start, gen_sequence.shape[-1]):
        current = gen_sequence[:, :, prev_offset:offset]
        next_token = model.sample_next(
            current,
            conditions,
            top_k=top_k,
            temperature=temperature,
            sampled_token_pool=sampled_token_pool,
        )
        mx.eval(next_token)
        sampled_token_pool.append(np.array(next_token[0, :, 0]))
        valid = mask[:, offset : offset + 1][None, :, :]
        next_token = mx.where(valid, next_token, model.config.special_token_id)
        next_token = mx.where(is_end, model.config.special_token_id, next_token)
        is_end = mx.logical_or(is_end, next_token == model.config.eos_token_id)
        current_slot = gen_sequence[:, :, offset : offset + 1]
        updated = mx.where(current_slot == -1, next_token, current_slot)
        gen_sequence = mx.slice_update(gen_sequence, updated, mx.array([0, 0, offset], dtype=mx.int32), axes=[0, 1, 2])
        prev_offset = offset
        if offset % 25 == 0:
            mx.eval(gen_sequence)
            print(f"{offset}/{gen_sequence.shape[-1] - 1}")
    output_sequence = mx.slice_update(output_sequence, gen_sequence, mx.array([0, 0, 0], dtype=mx.int32), axes=[0, 1, 2])
    tokens = revert_pattern_sequence(output_sequence, timesteps, special_token=-1)
    mx.eval(tokens)
    return tokens
