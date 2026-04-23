#!/usr/bin/env python3
"""
Precompute YaRN RoPE cosines/sines for a lalamo-converted classifier whose
safetensors file was emitted *without* rope tables (e.g.
openai/privacy-filter). Writes a sibling `model.with_rope.safetensors` (or
rewrites in-place with --in-place).

The math matches lalamo/lalamo/modules/rope.py::YARNRoPEConfig._scale_inverse_frequencies
byte-for-byte so the resulting tensors are indistinguishable from what lalamo
would have produced if it had persisted them.

Usage:
  python inject_rope_tables.py /path/to/model/dir           # writes sibling
  python inject_rope_tables.py /path/to/model/dir --in-place

Currently only YARNRoPEConfig is implemented because that's what
openai/privacy-filter uses. Other scaling kinds are trivial to add.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

try:
    from safetensors.numpy import load_file, save_file
except ImportError:
    print("This script requires `safetensors` (pip install safetensors).", file=sys.stderr)
    raise


def find_correction_dim(num_rotations: float, dim: int, base: float, original_context_length: int) -> float:
    return (dim * math.log(original_context_length / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    original_context_length: int,
    truncate: bool,
) -> tuple[float, float]:
    low = find_correction_dim(low_rot, dim, base, original_context_length)
    high = find_correction_dim(high_rot, dim, base, original_context_length)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0.0), min(high, float(dim - 1))


def linear_ramp_factor(min_value: float, max_value: float, dim: int) -> np.ndarray:
    if min_value == max_value:
        max_value += 0.001
    linear = (np.arange(dim, dtype=np.float32) - np.float32(min_value)) / (
        np.float32(max_value) - np.float32(min_value)
    )
    return np.clip(linear, 0.0, 1.0)


def yarn_tables(
    head_dim: int,
    max_sequence_length: int,
    base: float,
    scaling_factor: float,
    original_context_length: int,
    beta_fast: float,
    beta_slow: float,
    truncate: bool,
    partial_rotary_dim: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    timesteps = np.arange(max_sequence_length, dtype=np.float32)
    channel_indices = np.arange(0, head_dim, 2, dtype=np.int32)
    inv_freq = 1.0 / (base ** (channel_indices.astype(np.float32) / head_dim))

    # YaRN scaling
    scaled = inv_freq / scaling_factor
    low, high = find_correction_range(beta_fast, beta_slow, head_dim, base, original_context_length, truncate)
    smoothing = 1.0 - linear_ramp_factor(low, high, head_dim // 2)
    inv_freq = scaled * (1.0 - smoothing) + inv_freq * smoothing

    if partial_rotary_dim is not None and partial_rotary_dim < head_dim:
        rope_angles = partial_rotary_dim // 2
        mask = np.arange(head_dim // 2) < rope_angles
        inv_freq = inv_freq * mask

    outer = np.outer(timesteps, inv_freq)  # [T, head_dim/2]
    embeddings = np.concatenate([outer, outer], axis=-1)  # [T, head_dim]
    attn_scale = 0.1 * math.log(scaling_factor) + 1.0
    cos = (np.cos(embeddings) * attn_scale).astype(np.float32)
    sin = (np.sin(embeddings) * attn_scale).astype(np.float32)
    return cos, sin


def resolve_rope_cfg(model_config: dict) -> dict:
    """Grab the rope config from either transformer-level or layer[0]."""
    tf = model_config["transformer_config"]
    for key in ("global_rope_config", "local_rope_config"):
        if tf.get(key):
            return tf[key]
    first_layer = tf["layer_configs"][0]
    if first_layer.get("rope_config"):
        return first_layer["rope_config"]
    raise SystemExit("No rope config found in transformer_config or layer_configs[0].rope_config")


def bfloat16_from_float32(x: np.ndarray) -> np.ndarray:
    """
    numpy has no native bfloat16. Emulate: take the upper 16 bits of the
    float32 representation (round-to-nearest-even via adding 0x8000 before
    truncation).
    """
    f32 = x.astype(np.float32).view(np.uint32)
    # round to nearest even
    rounded = (f32 + 0x7FFF + ((f32 >> 16) & 1)) & 0xFFFF0000
    bf16_bits = (rounded >> 16).astype(np.uint16)
    return bf16_bits  # caller will write as BF16 via safetensors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--in-place", action="store_true", help="Overwrite model.safetensors instead of writing sibling")
    args = ap.parse_args()

    model_dir: Path = args.model_dir
    cfg_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not cfg_path.exists() or not weights_path.exists():
        raise SystemExit(f"Missing config.json or model.safetensors in {model_dir}")

    with cfg_path.open() as f:
        full_cfg = json.load(f)
    model_config = full_cfg["model_config"]["model_config"]
    rope_cfg = resolve_rope_cfg(model_config)
    if rope_cfg.get("type") != "YARNRoPEConfig":
        raise SystemExit(f"Only YARNRoPEConfig is handled here; got {rope_cfg.get('type')!r}")

    head_dim = rope_cfg["head_dim"]
    max_seq = model_config["context_length"]  # uzu's shared rope is sized to context_length
    base = float(rope_cfg["base"])
    scaling_factor = float(rope_cfg["scaling_factor"])
    original_context_length = int(rope_cfg["original_context_length"])
    beta_fast = float(rope_cfg["beta_fast"])
    beta_slow = float(rope_cfg["beta_slow"])
    truncate = bool(rope_cfg.get("truncate", False))
    partial_rotary_dim = rope_cfg.get("partial_rotary_dim")

    print(
        f"YaRN: head_dim={head_dim} max_seq={max_seq} base={base} "
        f"scaling={scaling_factor} orig_ctx={original_context_length} "
        f"beta_fast={beta_fast} beta_slow={beta_slow} truncate={truncate}"
    )

    cos_f32, sin_f32 = yarn_tables(
        head_dim=head_dim,
        max_sequence_length=max_seq,
        base=base,
        scaling_factor=scaling_factor,
        original_context_length=original_context_length,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        truncate=truncate,
        partial_rotary_dim=partial_rotary_dim,
    )

    print(f"cos shape={cos_f32.shape} sin shape={sin_f32.shape}")

    # The privacy-filter safetensors use bfloat16 throughout, which numpy
    # doesn't understand. Use torch for the round-trip (safetensors.torch
    # supports bf16).
    import torch
    from safetensors.torch import save_file as torch_save_file
    from safetensors.torch import load_file as torch_load_file

    torch_tensors = torch_load_file(str(weights_path))
    if "transformer.global_rope.cosines" in torch_tensors:
        print("transformer.global_rope.{cosines,sines} already present; overwriting.")

    cos_bf16 = torch.from_numpy(cos_f32).to(torch.bfloat16).contiguous()
    sin_bf16 = torch.from_numpy(sin_f32).to(torch.bfloat16).contiguous()
    torch_tensors["transformer.global_rope.cosines"] = cos_bf16
    torch_tensors["transformer.global_rope.sines"] = sin_bf16

    out_path = weights_path if args.in_place else weights_path.with_suffix(".with_rope.safetensors")
    torch_save_file(torch_tensors, str(out_path))
    print(f"Wrote {out_path} (bf16)")


if __name__ == "__main__":
    main()
