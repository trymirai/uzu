#!/usr/bin/env python3
"""Convert HuggingFace Haru-4B (BF16) into a mixed-precision uzu bundle.

Produces:
  - config.json (lalamo 0.19 nested schema for the Qwen3.5 hybrid LM backbone)
  - model.safetensors (decoder backbone: 8-bit MLX embedding/mixers, 3-bit JANGTQ or 4-bit MLX MLPs, F32 norms/conv)
  - haru_config.json (Haru duplex, MeanFlowHead, VibeVoiceCodec, and DRY parameters)
  - haru_sidecar.safetensors (audio projectors, FP32 PAD/EOP rows, MeanFlowHead, VibeVoiceCodec)
  - tokenizer.json, tokenizer_config.json, assets/

Uses a two-pass streaming safetensors writer (zero duplicate temporary files on disk).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

GROUP_SIZE = 64
JANGTQ3_BITS = 3
JANGTQ3_ALPHA = 0.47125
JANGTQ3_BETA = 0.0116

# Precompute 32x32 normalized Sylvester Hadamard matrix matching JangtqHadamard32.
def _make_hadamard32() -> np.ndarray:
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    for _ in range(4):
        h = np.kron(h, np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32))
    return (h / np.sqrt(32.0)).astype(np.float32)


H32 = _make_hadamard32()


def hadamard32_transform_rows(w_f32: np.ndarray) -> np.ndarray:
    """Apply blockwise 32x32 orthonormal Walsh-Hadamard transform along columns of [N, K]."""
    n, k = w_f32.shape
    assert k % 32 == 0, f"K={k} must be a multiple of 32 for Hadamard32"
    return (w_f32.reshape(n, k // 32, 32) @ H32).reshape(n, k)
JANGTQ3_LEVELS = (
    (np.arange(8, dtype=np.float32) - 3.5)
    * (JANGTQ3_ALPHA + JANGTQ3_BETA * (np.arange(8, dtype=np.float32) - 3.5) ** 2)
).astype(np.float32)
JANGTQ3_MIDPOINTS = (0.5 * (JANGTQ3_LEVELS[:-1] + JANGTQ3_LEVELS[1:])).astype(np.float32)
JANGTQ3_LEVELS_RMS = float(np.sqrt(np.mean(JANGTQ3_LEVELS ** 2)))


def bf16_bytes_to_f32(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    # Handle NaN/Inf safely + IEEE round-to-nearest-even
    rounding_bias = ((u32 >> 16) & 1) + np.uint32(0x7FFF)
    u16 = ((u32 + rounding_bias) >> 16).astype("<u2")
    return u16.tobytes()


def f32_to_bf16_roundtrip(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    rounding_bias = ((u32 >> 16) & 1) + np.uint32(0x7FFF)
    truncated = (u32 + rounding_bias) & np.uint32(0xFFFF0000)
    return truncated.view(np.float32)


def quantize_mlx_affine(
    w_f32: np.ndarray,
    bits: int,
    layout: str,
    group_size: int = GROUP_SIZE,
) -> tuple[bytes, bytes, bytes]:
    """Quantize [N, K] float32 weights into (weights_bytes, scales_bf16_bytes, biases_bf16_bytes)."""
    n, k = w_f32.shape
    assert k % group_size == 0, f"K={k} not divisible by group_size={group_size}"
    g = k // group_size
    max_q = (1 << bits) - 1
    wg = w_f32.reshape(n, g, group_size)
    w_min = wg.min(axis=-1)
    w_max = wg.max(axis=-1)

    scales = (w_max - w_min) / float(max_q)
    scales = np.where(scales == 0.0, 1.0, scales).astype(np.float32)
    biases = w_min.astype(np.float32)

    # Round scales & biases to BF16 first so codes optimize against exact stored params.
    scales_q = f32_to_bf16_roundtrip(scales)
    scales_q = np.where(scales_q == 0.0, 1.0, scales_q)
    biases_q = f32_to_bf16_roundtrip(biases)

    codes = np.rint((wg - biases_q[..., None]) / scales_q[..., None])
    codes = np.clip(codes, 0, max_q).astype(np.uint8).reshape(n, k)

    if bits == 8:
        w_bytes = np.ascontiguousarray(codes).tobytes()
    elif bits == 4:
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        w_bytes = np.ascontiguousarray(packed, dtype=np.uint8).tobytes()
    else:
        raise ValueError(f"unsupported MLX bits={bits}")

    if layout == "input_output":
        s_out = scales_q
        b_out = biases_q
    elif layout == "output_input":
        s_out = np.ascontiguousarray(scales_q.T)
        b_out = np.ascontiguousarray(biases_q.T)
    else:
        raise ValueError(f"unsupported layout={layout}")

    return w_bytes, f32_to_bf16_bytes(s_out), f32_to_bf16_bytes(b_out)


def pack_3bit_lsb_u32(codes: np.ndarray) -> bytes:
    """Pack [N, K] uint8 codes in [0..7] into LSB-first U32 words [N, K * 3 // 32]."""
    n, k = codes.shape
    assert k % 32 == 0, f"K={k} must be a multiple of 32 for 3-bit packing"
    c = codes.astype(np.uint32).reshape(n, k // 8, 8)
    v24 = (
        c[:, :, 0]
        | (c[:, :, 1] << 3)
        | (c[:, :, 2] << 6)
        | (c[:, :, 3] << 9)
        | (c[:, :, 4] << 12)
        | (c[:, :, 5] << 15)
        | (c[:, :, 6] << 18)
        | (c[:, :, 7] << 21)
    )
    out = np.empty((n, k // 8, 3), dtype=np.uint8)
    out[:, :, 0] = (v24 & 0xFF).astype(np.uint8)
    out[:, :, 1] = ((v24 >> 8) & 0xFF).astype(np.uint8)
    out[:, :, 2] = ((v24 >> 16) & 0xFF).astype(np.uint8)
    # View flat row bytes as little-endian u32 [N, K * 3 // 32]
    words = np.ascontiguousarray(out).reshape(n, (k * 3) // 8).view("<u4")
    return words.tobytes()


def quantize_jangtq3(w_f32: np.ndarray, iters: int = 4) -> tuple[bytes, bytes]:
    """Rotate [N, K] via Hadamard32 on K and quantize per-row to 3-bit JANGTQ."""
    n, k = w_f32.shape
    assert k % 512 == 0 and n % 32 == 0, f"JANGTQ expects K%512==0 and N%32==0, got ({n}, {k})"
    w_rot = hadamard32_transform_rows(w_f32)

    # Initial RMS-matched scale
    rms = np.sqrt(np.mean(w_rot * w_rot, axis=1))
    s = np.maximum(rms / JANGTQ3_LEVELS_RMS, 1e-8).astype(np.float32)

    for _ in range(iters):
        normed = w_rot / s[:, None]
        codes = np.digitize(normed, JANGTQ3_MIDPOINTS).astype(np.uint8)
        q = JANGTQ3_LEVELS[codes]
        denom = np.sum(q * q, axis=1)
        numer = np.sum(w_rot * q, axis=1)
        s = np.where(denom > 1e-12, numer / np.maximum(denom, 1e-12), s)
        s = np.maximum(s, 1e-8).astype(np.float32)

    # Snap scales to stored F16 and do final assignment against F16 scale.
    s_f16 = s.astype("<f2")
    s_stored = np.maximum(s_f16.astype(np.float32), 1e-8)
    codes = np.digitize(w_rot / s_stored[:, None], JANGTQ3_MIDPOINTS).astype(np.uint8)
    return pack_3bit_lsb_u32(codes), s_f16.tobytes()


class SafetensorsSource:
    def __init__(self, path: Path):
        self.path = path
        self.file = open(path, "rb")
        (hlen,) = struct.unpack("<Q", self.file.read(8))
        self.header = json.loads(self.file.read(hlen))
        self.data_start = 8 + hlen

    def info(self, name: str) -> dict:
        return self.header[name]

    def raw(self, name: str) -> bytes:
        info = self.header[name]
        b0, b1 = info["data_offsets"]
        self.file.seek(self.data_start + b0)
        return self.file.read(b1 - b0)

    def f32(self, name: str) -> np.ndarray:
        info = self.header[name]
        dtype = info["dtype"]
        shape = tuple(info["shape"])
        raw = self.raw(name)
        if dtype == "BF16":
            return bf16_bytes_to_f32(raw, shape)
        if dtype == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
        if dtype == "F16":
            return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
        raise ValueError(f"unexpected dtype {dtype} for {name}")

    def close(self) -> None:
        self.file.close()


DTYPE_BYTES = {
    "U8": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
}


@dataclass
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    producer: Callable[[], bytes]

    @property
    def nbytes(self) -> int:
        elems = 1
        for d in self.shape:
            elems *= d
        return elems * DTYPE_BYTES[self.dtype]


class StreamingSafetensorsBuilder:
    def __init__(self) -> None:
        self.entries: list[TensorEntry] = []
        self.metadata: dict[str, str] = {}

    def add(self, name: str, dtype: str, shape: tuple[int, ...] | list[int], producer: Callable[[], bytes]) -> None:
        self.entries.append(TensorEntry(name=name, dtype=dtype, shape=tuple(int(x) for x in shape), producer=producer))

    def set_spec(self, key: str, spec: dict) -> None:
        self.metadata[key] = json.dumps(spec, separators=(",", ":"))

    def write(self, dst_path: Path, progress_label: str) -> int:
        header: dict = {}
        if self.metadata:
            header["__metadata__"] = self.metadata
        offset = 0
        for entry in self.entries:
            end = offset + entry.nbytes
            header[entry.name] = {
                "dtype": entry.dtype,
                "shape": list(entry.shape),
                "data_offsets": [offset, end],
            }
            offset = end

        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        t0 = time.perf_counter()
        with open(dst_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            for idx, entry in enumerate(self.entries):
                data = entry.producer()
                if len(data) != entry.nbytes:
                    raise RuntimeError(
                        f"{entry.name}: expected {entry.nbytes} bytes ({entry.dtype} {entry.shape}), got {len(data)}"
                    )
                f.write(data)
                if (idx + 1) % 64 == 0 or idx + 1 == len(self.entries):
                    dt = time.perf_counter() - t0
                    pct = 100.0 * (idx + 1) / len(self.entries)
                    print(f"  [{progress_label}] {idx + 1}/{len(self.entries)} tensors ({pct:.0f}%, {dt:.1f}s)", flush=True)
                    gc.collect()
        return 8 + len(header_bytes) + offset


def build_uzu_config(haru_cfg: dict) -> dict:
    lm = haru_cfg["lm"]
    hidden = lm["hidden_size"]
    inter = lm["intermediate_size"]
    eps = lm["rms_norm_eps"]
    layer_types = lm["layer_types"]
    max_seq = lm["max_position_embeddings"]

    def norm_cfg(full_layer: bool = True) -> dict:
        return {
            "epsilon": eps,
            "scale_offset": None,
            "upcast_mode": "full_layer" if full_layer else "only_normalization",
            "subtract_mean": False,
            "has_scale": True,
            "has_biases": False,
        }

    layer_configs = []
    for ltype in layer_types:
        is_attn = ltype == "full_attention"
        if is_attn:
            mixer_body = {
                "type": "AttentionConfig",
                "qkvg_projection_config": {},
                "out_projection_config": {},
                "query_norm_config": norm_cfg(full_layer=True),
                "key_norm_config": norm_cfg(full_layer=True),
                "num_heads": lm["num_attention_heads"],
                "num_groups": lm["num_key_value_heads"],
                "head_dim": lm["head_dim"],
                "is_causal": True,
                "scale": None,
                "sliding_window_size": None,
                "logit_soft_cap": None,
                "has_sinks": False,
                "has_qkvg_biases": False,
                "has_out_biases": False,
                "normalize_values": False,
                "is_kv_sharing": False,
                "has_gate": bool(lm.get("attn_output_gate", True)),
            }
            rotary_dim = int(lm["head_dim"] * lm.get("partial_rotary_factor", 0.25))
            rope_cfg = {
                "type": "UnscaledRoPEConfig",
                "base": float(lm["rope_parameters"]["rope_theta"]),
                "max_sequence_length": max_seq,
                "head_dim": rotary_dim,
            }
        else:
            mixer_body = {
                "type": "DeltaNetConfig",
                "in_proj_config": {},
                "conv_config": {"has_biases": False},
                "out_proj_config": {},
                "norm_config": norm_cfg(full_layer=False),
                "num_heads": lm["linear_num_value_heads"],
                "num_groups": lm["linear_num_key_heads"],
                "head_dim": lm["linear_key_head_dim"],
                "value_head_dim": lm["linear_value_head_dim"],
                "kernel_size": lm["linear_conv_kernel_dim"],
                "output_gate": {"type": "SiLU", "alpha": 1.0},
            }
            rope_cfg = None

        layer_configs.append(
            {
                "prologue_configs": [],
                "mixer": {
                    "conv_config": None,
                    "pre_norm_config": norm_cfg(full_layer=True),
                    "body_config": mixer_body,
                    "post_norm_config": None,
                },
                "mlp": {
                    "conv_config": None,
                    "pre_norm_config": norm_cfg(full_layer=True),
                    "body_config": {
                        "type": "DenseMLPConfig",
                        "linear_config": {},
                        "activation": {"type": "SiLU", "alpha": 1.0},
                        "has_up_biases": False,
                        "has_down_biases": False,
                        "gate_clipping": None,
                        "up_clipping": None,
                    },
                    "post_norm_config": None,
                },
                "epilogue_configs": [],
                "hidden_dim": None,
                "kv_source_layer_index": None,
                "rope_config": rope_cfg,
            }
        )

    return {
        "type": "LanguageModelConfig",
        "decoder_config": {
            "embedding_config": {
                "type": "TiedEmbeddingConfig",
                "input_scale": None,
                "logit_scale": None,
                "logit_soft_cap": None,
                "permute_indices": False,
            },
            "transformer_config": {
                "residual_config": {"type": "AdditiveResidualConfig"},
                "layer_configs": layer_configs,
                "output_norm_config": norm_cfg(full_layer=True),
                "model_dim": hidden,
                "hidden_dim": inter,
            },
            "vocab_size": lm["vocab_size"],
            "ple_model_config": None,
            "embedding_norm_config": None,
        },
        "generation_config": {
            "stop_token_ids": [lm["eos_token_id"]] if lm.get("eos_token_id") is not None else [],
            "temperature": 0.7,
            "top_k": 25,
            "top_p": 1.0,
            "min_p": None,
            "banned_tokens": None,
            "repetition_penalty": None,
            "presence_penalty": None,
            "frequency_penalty": None,
            "suffix_repetition_length": None,
        },
        "token_codec_config": {
            "type": "RawTextCodecConfig",
        },
    }


def add_mlx_quant_linear(
    builder: StreamingSafetensorsBuilder,
    dst_prefix: str,
    n: int,
    k: int,
    bits: int,
    layout: str,
    weight_loader: Callable[[], np.ndarray],
    leaf: str = "weights",
) -> None:
    """Register lazy MLXSpec quantized tensors for a single [N, K] matrix."""
    assert k % GROUP_SIZE == 0, f"{dst_prefix}: K={k} not multiple of {GROUP_SIZE}"
    g = k // GROUP_SIZE
    cache: dict[str, bytes] = {}

    def ensure() -> None:
        if "w" not in cache:
            w = weight_loader()
            assert w.shape == (n, k), f"{dst_prefix}: expected {(n, k)}, got {w.shape}"
            wb, sb, bb = quantize_mlx_affine(w, bits=bits, layout=layout, group_size=GROUP_SIZE)
            cache["w"] = wb
            cache["s"] = sb
            cache["b"] = bb

    w_cols = k if bits == 8 else k // 2
    sb_shape = (n, g) if layout == "input_output" else (g, n)
    prefix = f"{dst_prefix}.{leaf}" if leaf else dst_prefix

    def pop(key: str) -> bytes:
        ensure()
        return cache.pop(key)

    builder.add(f"{prefix}.weights", "U8", (n, w_cols), lambda: pop("w"))
    builder.add(f"{prefix}.scales", "BF16", sb_shape, lambda: pop("s"))
    builder.add(f"{prefix}.biases", "BF16", sb_shape, lambda: pop("b"))
    builder.set_spec(
        f"{prefix}.spec",
        {"type": "MLXSpec", "bits": bits, "group_size": GROUP_SIZE, "layout": layout},
    )


def add_hybrid_hadamard_mlx_linear(
    builder: StreamingSafetensorsBuilder,
    dst_prefix: str,
    n: int,
    k: int,
    bits: int,
    layout: str,
    weight_loader: Callable[[], np.ndarray],
    leaf: str = "weights",
) -> None:
    """Register lazy HybridSpec(HadamardSpec(32, input) + MLXSpec) tensors for a single [N, K] matrix."""
    assert k % 32 == 0 and k % GROUP_SIZE == 0, f"{dst_prefix}: K={k} must be divisible by 32 and {GROUP_SIZE}"
    g = k // GROUP_SIZE
    cache: dict[str, bytes] = {}

    def ensure() -> None:
        if "w" not in cache:
            w = weight_loader()
            assert w.shape == (n, k), f"{dst_prefix}: expected {(n, k)}, got {w.shape}"
            w_rot = hadamard32_transform_rows(w)
            wb, sb, bb = quantize_mlx_affine(w_rot, bits=bits, layout=layout, group_size=GROUP_SIZE)
            cache["w"] = wb
            cache["s"] = sb
            cache["b"] = bb

    w_cols = k if bits == 8 else k // 2
    sb_shape = (n, g) if layout == "input_output" else (g, n)
    prefix = f"{dst_prefix}.{leaf}" if leaf else dst_prefix

    def pop(key: str) -> bytes:
        ensure()
        return cache.pop(key)

    builder.add(
        f"{prefix}.incoherence_signs.input_signs",
        "I32",
        (k,),
        lambda k=k: np.ones((k,), dtype="<i4").tobytes(),
    )
    builder.add(f"{prefix}.quantized.weights", "U8", (n, w_cols), lambda: pop("w"))
    builder.add(f"{prefix}.quantized.scales", "BF16", sb_shape, lambda: pop("s"))
    builder.add(f"{prefix}.quantized.biases", "BF16", sb_shape, lambda: pop("b"))
    builder.set_spec(
        f"{prefix}.spec",
        {
            "type": "HybridSpec",
            "quantization_spec": {
                "type": "MLXSpec",
                "bits": bits,
                "group_size": GROUP_SIZE,
                "layout": layout,
            },
            "adapter_spec": None,
            "incoherence_spec": {
                "type": "HadamardSpec",
                "block_size": 32,
                "processing_mode": "input",
            },
        },
    )


def add_jangtq3_linear(
    builder: StreamingSafetensorsBuilder,
    dst_prefix: str,
    n: int,
    k: int,
    weight_loader: Callable[[], np.ndarray],
) -> None:
    """Register lazy Jangtq2Spec(3-bit) tensors under <dst_prefix>.weights.quantized.{weights,scales}."""
    assert k % 512 == 0 and n % 32 == 0, f"{dst_prefix}: shape {(n, k)} incompatible with JANGTQ"
    words = (k * JANGTQ3_BITS) // 32
    cache: dict[str, bytes] = {}

    def ensure() -> None:
        if "w" not in cache:
            w = weight_loader()
            assert w.shape == (n, k), f"{dst_prefix}: expected {(n, k)}, got {w.shape}"
            wb, sb = quantize_jangtq3(w)
            cache["w"] = wb
            cache["s"] = sb

    def pop(key: str) -> bytes:
        ensure()
        return cache.pop(key)

    builder.add(f"{dst_prefix}.weights.quantized.weights", "U32", (n, words), lambda: pop("w"))
    builder.add(f"{dst_prefix}.weights.quantized.scales", "F16", (n,), lambda: pop("s"))
    builder.set_spec(
        f"{dst_prefix}.weights.spec",
        {"type": "Jangtq2Spec", "bits": JANGTQ3_BITS, "alpha": JANGTQ3_ALPHA, "beta": JANGTQ3_BETA},
    )


def verify_jangtq3_packing() -> None:
    """Self-test pack_3bit_lsb_u32 against scalar bit shifts matching jangtq.metal."""
    rng = np.random.default_rng(42)
    codes = rng.integers(0, 8, size=(4, 512), dtype=np.uint8)
    packed_bytes = pack_3bit_lsb_u32(codes)
    words = np.frombuffer(packed_bytes, dtype="<u4").reshape(4, 48)
    for r in range(4):
        for c in range(512):
            bit = c * 3
            qbase = bit >> 5
            qshift = bit & 31
            lo = int(words[r, qbase])
            hi = int(words[r, qbase + 1]) if qbase + 1 < 48 else 0
            extracted = ((lo | (hi << 32)) >> qshift) & 7
            assert extracted == int(codes[r, c]), f"pack mismatch at ({r}, {c}): {extracted} != {codes[r, c]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--mlp-format", choices=("hybrid4", "mlx4", "mlx8", "jangtq3"), default="mlx8")
    parser.add_argument("--embed-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sidecar-only", action="store_true")
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()

    verify_jangtq3_packing()

    src_dir: Path = args.src.expanduser()
    dst_dir: Path = args.dst.expanduser()
    if dst_dir.exists():
        if args.sidecar_only:
            if not (dst_dir / "model.safetensors").exists():
                sys.exit(f"--sidecar-only requires existing {dst_dir / 'model.safetensors'}")
        elif args.model_only:
            if not (dst_dir / "haru_sidecar.safetensors").exists():
                sys.exit(f"--model-only requires existing {dst_dir / 'haru_sidecar.safetensors'}")
            (dst_dir / "model.safetensors").unlink(missing_ok=True)
        elif not args.overwrite:
            sys.exit(f"Refusing to overwrite {dst_dir}; pass --overwrite")
        else:
            shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    haru_cfg = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    lm = haru_cfg["lm"]
    codec_cfg = haru_cfg["codec"]
    head_cfg = haru_cfg["head"]
    src = SafetensorsSource(src_dir / "model.safetensors")

    hidden = lm["hidden_size"]
    inter = lm["intermediate_size"]
    vocab = lm["vocab_size"]
    pad_id = haru_cfg["pad_token_id"]
    eop_id = haru_cfg["eop_token_id"]

    q_heads = lm["num_attention_heads"]
    kv_heads = lm["num_key_value_heads"]
    attn_hd = lm["head_dim"]
    q_dim = q_heads * attn_hd
    kv_dim = kv_heads * attn_hd

    gdn_k_heads = lm["linear_num_key_heads"]
    gdn_v_heads = lm["linear_num_value_heads"]
    gdn_k_dim = lm["linear_key_head_dim"]
    gdn_v_dim = lm["linear_value_head_dim"]
    gdn_conv_dim = 2 * gdn_k_heads * gdn_k_dim + gdn_v_heads * gdn_v_dim
    gdn_val_dim = gdn_v_heads * gdn_v_dim
    gdn_total_in = gdn_conv_dim + gdn_val_dim + 2 * gdn_v_heads

    consumed: set[str] = set()

    def read_f32(name: str) -> np.ndarray:
        consumed.add(name)
        return src.f32(name)

    def read_raw(name: str) -> bytes:
        consumed.add(name)
        return src.raw(name)

    def make_gamma_bytes(name: str) -> bytes:
        return (read_f32(name) + 1.0).astype("<f4").tobytes()

    def make_f32_bytes(name: str) -> bytes:
        return read_f32(name).astype("<f4").tobytes()

    # =========================================================================
    # 1. Main backbone safetensors (`model.safetensors`)
    # =========================================================================
    main_b = StreamingSafetensorsBuilder()

    # Tied embedding (`input_output`, 8-bit group 64)
    add_mlx_quant_linear(
        main_b,
        "decoder.embedding",
        vocab,
        hidden,
        bits=args.embed_bits,
        layout="input_output",
        weight_loader=lambda: read_f32("lm.embed_tokens.weight"),
        leaf="embedding",
    )

    main_b.add(
        "decoder.transformer.output_norm.scales",
        "F32",
        (hidden,),
        lambda: make_gamma_bytes("lm.norm.weight"),
    )

    for i, ltype in enumerate(lm["layer_types"]):
        lp = f"lm.layers.{i}"
        up = f"decoder.transformer.layers.{i}"
        use_hybrid = args.mlp_format == "hybrid4"
        hybrid_bits = 8 if i == 0 else 4
        add_backbone_linear = add_hybrid_hadamard_mlx_linear if use_hybrid else add_mlx_quant_linear
        mixer_bits = 8

        main_b.add(
            f"{up}.mixer.pre_norm.scales",
            "F32",
            (hidden,),
            lambda lp=lp: make_gamma_bytes(f"{lp}.input_layernorm.weight"),
        )
        main_b.add(
            f"{up}.mlp.pre_norm.scales",
            "F32",
            (hidden,),
            lambda lp=lp: make_gamma_bytes(f"{lp}.post_attention_layernorm.weight"),
        )

        if ltype == "full_attention":
            def load_qkvg(lp: str = lp) -> np.ndarray:
                q_and_gate = read_f32(f"{lp}.self_attn.q_proj.weight").reshape(q_heads, 2, attn_hd, hidden)
                q = q_and_gate[:, 0, :, :].reshape(q_dim, hidden)
                gate = q_and_gate[:, 1, :, :].reshape(q_dim, hidden)
                k = read_f32(f"{lp}.self_attn.k_proj.weight")
                v = read_f32(f"{lp}.self_attn.v_proj.weight")
                return np.concatenate([q, k, v, gate], axis=0)

            add_backbone_linear(
                main_b,
                f"{up}.mixer.body.qkvg_projection",
                2 * q_dim + 2 * kv_dim,
                hidden,
                bits=mixer_bits,
                layout="output_input",
                weight_loader=load_qkvg,
            )
            add_backbone_linear(
                main_b,
                f"{up}.mixer.body.out_projection",
                hidden,
                q_dim,
                bits=mixer_bits,
                layout="output_input",
                weight_loader=lambda lp=lp: read_f32(f"{lp}.self_attn.o_proj.weight"),
            )
            main_b.add(
                f"{up}.mixer.body.query_norm.scales",
                "F32",
                (attn_hd,),
                lambda lp=lp: make_gamma_bytes(f"{lp}.self_attn.q_norm.weight"),
            )
            main_b.add(
                f"{up}.mixer.body.key_norm.scales",
                "F32",
                (attn_hd,),
                lambda lp=lp: make_gamma_bytes(f"{lp}.self_attn.k_norm.weight"),
            )
        else:
            def load_gdn_in(lp: str = lp) -> np.ndarray:
                return np.concatenate(
                    [
                        read_f32(f"{lp}.linear_attn.in_proj_qkv.weight"),
                        read_f32(f"{lp}.linear_attn.in_proj_z.weight"),
                        read_f32(f"{lp}.linear_attn.in_proj_b.weight"),
                        read_f32(f"{lp}.linear_attn.in_proj_a.weight"),
                    ],
                    axis=0,
                )

            add_backbone_linear(
                main_b,
                f"{up}.mixer.body.in_proj",
                gdn_total_in,
                hidden,
                bits=mixer_bits,
                layout="output_input",
                weight_loader=load_gdn_in,
            )
            add_backbone_linear(
                main_b,
                f"{up}.mixer.body.out_proj",
                hidden,
                gdn_val_dim,
                bits=mixer_bits,
                layout="output_input",
                weight_loader=lambda lp=lp: read_f32(f"{lp}.linear_attn.out_proj.weight"),
            )
            main_b.add(
                f"{up}.mixer.body.conv.weights",
                "F32",
                (gdn_conv_dim, lm["linear_conv_kernel_dim"]),
                lambda lp=lp: read_f32(f"{lp}.linear_attn.conv1d.weight")
                .reshape(gdn_conv_dim, lm["linear_conv_kernel_dim"])
                .astype("<f4")
                .tobytes(),
            )
            main_b.add(
                f"{up}.mixer.body.a_log",
                "F32",
                (gdn_v_heads,),
                lambda lp=lp: make_f32_bytes(f"{lp}.linear_attn.A_log"),
            )
            main_b.add(
                f"{up}.mixer.body.dt_bias",
                "F32",
                (gdn_v_heads,),
                lambda lp=lp: make_f32_bytes(f"{lp}.linear_attn.dt_bias"),
            )
            # Standard RMSNorm gain (not zero-centered!)
            main_b.add(
                f"{up}.mixer.body.norm.scales",
                "F32",
                (gdn_v_dim,),
                lambda lp=lp: make_f32_bytes(f"{lp}.linear_attn.norm.weight"),
            )

        if args.mlp_format == "jangtq3":
            add_jangtq3_linear(
                main_b,
                f"{up}.mlp.body.gate_projection",
                inter,
                hidden,
                weight_loader=lambda lp=lp: read_f32(f"{lp}.mlp.gate_proj.weight"),
            )
            add_jangtq3_linear(
                main_b,
                f"{up}.mlp.body.up_projection",
                inter,
                hidden,
                weight_loader=lambda lp=lp: read_f32(f"{lp}.mlp.up_proj.weight"),
            )
            add_jangtq3_linear(
                main_b,
                f"{up}.mlp.body.down_projection",
                hidden,
                inter,
                weight_loader=lambda lp=lp: read_f32(f"{lp}.mlp.down_proj.weight"),
            )
        else:
            mlp_bits = 8 if args.mlp_format == "mlx8" else (hybrid_bits if use_hybrid else 4)
            # uzu GatedActMul expects [value(up), gate] order!
            add_backbone_linear(
                main_b,
                f"{up}.mlp.body.up_projection",
                2 * inter,
                hidden,
                bits=mlp_bits,
                layout="output_input",
                weight_loader=lambda lp=lp: np.concatenate(
                    [read_f32(f"{lp}.mlp.up_proj.weight"), read_f32(f"{lp}.mlp.gate_proj.weight")],
                    axis=0,
                ),
            )
            add_backbone_linear(
                main_b,
                f"{up}.mlp.body.down_projection",
                hidden,
                inter,
                bits=mlp_bits,
                layout="output_input",
                weight_loader=lambda lp=lp: read_f32(f"{lp}.mlp.down_proj.weight"),
            )

    # =========================================================================
    # 2. Sidecar safetensors (`haru_sidecar.safetensors`)
    # =========================================================================
    side_b = StreamingSafetensorsBuilder()

    # Exact FP32 PAD and EOP rows from unquantized embed_tokens
    side_b.add(
        "pad_row_fp32",
        "F32",
        (hidden,),
        lambda: read_f32("lm.embed_tokens.weight")[pad_id].astype("<f4").tobytes(),
    )
    side_b.add(
        "eop_row_fp32",
        "F32",
        (hidden,),
        lambda: read_f32("lm.embed_tokens.weight")[eop_id].astype("<f4").tobytes(),
    )

    for scalar_name in ("latent_scale", "latent_bias"):
        side_b.add(
            scalar_name,
            "F32",
            (1,),
            lambda k=scalar_name: read_f32(k).reshape(1).astype("<f4").tobytes(),
        )

    for proj in ("model_audio_proj", "user_audio_proj"):
        for sub in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
            k = f"{proj}.{sub}"
            side_b.add(k, "BF16", src.info(k)["shape"], lambda k=k: read_raw(k))
        nk = f"{proj}.norm.weight"
        side_b.add(nk, "F32", (hidden,), lambda nk=nk: make_f32_bytes(nk))

    # MeanFlowHead
    h_dim = head_cfg["hidden_size"]
    h_ffn = int(h_dim * head_cfg["ffn_ratio"])
    h_lat = head_cfg["latent_dim"]
    for k in (
        "head.latent_proj.weight",
        "head.cond_proj.weight",
        "head.final.proj.weight",
        "head.t_embed.fc1.weight",
        "head.t_embed.fc2.weight",
        "head.dt_embed.fc1.weight",
        "head.dt_embed.fc2.weight",
        "head.h_embed.fc1.weight",
        "head.h_embed.fc2.weight",
        "head.omega_embed.fc1.weight",
        "head.omega_embed.fc2.weight",
    ):
        side_b.add(k, "BF16", src.info(k)["shape"], lambda k=k: read_raw(k))

    def load_head_ada() -> np.ndarray:
        parts = [read_f32(f"head.layers.{i}.ada.weight") for i in range(head_cfg["num_layers"])]
        parts.append(read_f32("head.final.ada.weight"))
        return np.concatenate(parts, axis=0)

    ada_rows = (3 * head_cfg["num_layers"] + 2) * h_dim
    add_mlx_quant_linear(
        side_b,
        "head.ada",
        ada_rows,
        h_dim,
        bits=8,
        layout="output_input",
        weight_loader=load_head_ada,
        leaf="",
    )

    for i in range(head_cfg["num_layers"]):
        lp = f"head.layers.{i}"
        side_b.add(f"{lp}.norm.weight", "F32", (h_dim,), lambda lp=lp: make_f32_bytes(f"{lp}.norm.weight"))
        if args.mlp_format == "jangtq3":
            add_jangtq3_linear(
                side_b,
                f"{lp}.ffn.gate_projection",
                h_ffn,
                h_dim,
                weight_loader=lambda lp=lp: read_f32(f"{lp}.ffn.gate_proj.weight"),
            )
            add_jangtq3_linear(
                side_b,
                f"{lp}.ffn.up_projection",
                h_ffn,
                h_dim,
                weight_loader=lambda lp=lp: read_f32(f"{lp}.ffn.up_proj.weight"),
            )
            add_jangtq3_linear(
                side_b,
                f"{lp}.ffn.down_projection",
                h_dim,
                h_ffn,
                weight_loader=lambda lp=lp: read_f32(f"{lp}.ffn.down_proj.weight"),
            )
        else:
            head_bits = 8
            add_mlx_quant_linear(
                side_b,
                f"{lp}.ffn.up_projection",
                2 * h_ffn,
                h_dim,
                bits=head_bits,
                layout="output_input",
                weight_loader=lambda lp=lp: np.concatenate(
                    [read_f32(f"{lp}.ffn.up_proj.weight"), read_f32(f"{lp}.ffn.gate_proj.weight")],
                    axis=0,
                ),
            )
            add_mlx_quant_linear(
                side_b,
                f"{lp}.ffn.down_projection",
                h_dim,
                h_ffn,
                bits=head_bits,
                layout="output_input",
                weight_loader=lambda lp=lp: read_f32(f"{lp}.ffn.down_proj.weight"),
            )

    # VibeVoiceCodec: convert conv weights into matmul-ready layouts + keep encoder at 8-bit and decoder mixed 4/8-bit.
    def add_codec_matrix(
        weight_key: str,
        out_rows: int,
        in_cols: int,
        loader: Callable[[], np.ndarray],
        bf16_raw: bool = False,
    ) -> None:
        if weight_key.startswith("codec.encoder."):
            quant_ok = (out_rows >= 1024 or in_cols >= 1024) and (in_cols % GROUP_SIZE == 0) and in_cols >= 512 and (out_rows % 4 == 0)
            bits = 8
        else:
            quant_ok = out_rows >= 64 and in_cols >= 64 and (in_cols % GROUP_SIZE == 0) and (out_rows % 4 == 0)
            bits = 8

        if quant_ok:
            stem = weight_key[: -len(".weight")]
            add_mlx_quant_linear(side_b, stem, out_rows, in_cols, bits, "output_input", loader, leaf="")
        elif bf16_raw:
            side_b.add(weight_key, "BF16", (out_rows, in_cols), lambda k=weight_key: read_raw(k))
        else:
            side_b.add(
                weight_key,
                "BF16",
                (out_rows, in_cols),
                lambda loader=loader: f32_to_bf16_bytes(loader()),
            )

    codec_keys = sorted(k for k in src.header if k != "__metadata__" and k.startswith("codec."))
    for k in codec_keys:
        info = src.info(k)
        shape = tuple(info["shape"])
        if k.endswith(".conv.weight") and ".stages." in k:
            # Depthwise 7-tap filter [C, 1, 7] -> [C, 7] F32
            side_b.add(
                k,
                "F32",
                (shape[0], shape[2]),
                lambda k=k, s=shape: read_f32(k).reshape(s[0], s[2]).astype("<f4").tobytes(),
            )
        elif k.startswith("codec.decoder.up.") and k.endswith(".weight"):
            # ConvTranspose1d [in_c, out_c, K] -> [out_c * K, in_c] matching haru_mlx CONV_TRANSPOSE_MATRIX
            out_rows = shape[1] * shape[2]
            in_cols = shape[0]

            def load_ct(k: str = k, out_rows: int = out_rows, in_cols: int = in_cols) -> np.ndarray:
                return read_f32(k).transpose(1, 2, 0).reshape(out_rows, in_cols)

            add_codec_matrix(k, out_rows, in_cols, load_ct)
        elif len(shape) == 3 and k.endswith(".weight"):
            # CausalConv1d [out_c, in_c, K] -> [out_c, K * in_c] matching haru_mlx CONV_MATRIX
            out_rows = shape[0]
            in_cols = shape[2] * shape[1]

            def load_c(k: str = k, out_rows: int = out_rows, in_cols: int = in_cols) -> np.ndarray:
                return read_f32(k).transpose(0, 2, 1).reshape(out_rows, in_cols)

            add_codec_matrix(k, out_rows, in_cols, load_c)
        elif len(shape) == 2 and k.endswith(".weight"):
            out_rows, in_cols = shape
            add_codec_matrix(k, out_rows, in_cols, lambda k=k: read_f32(k), bf16_raw=True)
        else:
            # 1D norms, gammas, and biases -> F32 for clean accumulation on small vectors
            side_b.add(k, "F32", shape, lambda k=k: make_f32_bytes(k))

    print(f"Writing configs & streaming safetensors to {dst_dir} (mlp_format={args.mlp_format})...")
    uzu_cfg = build_uzu_config(haru_cfg)
    (dst_dir / "config.json").write_text(json.dumps(uzu_cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    haru_sidecar_cfg = {
        "model_type": "haru",
        "sample_rate": haru_cfg["sample_rate"],
        "frame_samples": haru_cfg["frame_samples"],
        "frame_rate": haru_cfg["frame_rate"],
        "latent_dim": haru_cfg["latent_dim"],
        "hidden_size": hidden,
        "pad_token_id": pad_id,
        "eop_token_id": eop_id,
        "eos_token_id": lm.get("eos_token_id"),
        "mlp_format": args.mlp_format,
        "codec": codec_cfg,
        "head": head_cfg,
        "defaults": {
            "guidance_scale": head_cfg.get("guidance_scale", 1.5),
            "temperature": 0.7,
            "top_k": 25,
            "top_p": 1.0,
            "eop_bias": 0.0,
            "dry": {
                "multiplier": 0.8,
                "base": 1.75,
                "allowed_length": 1,
                "lookback": 1024,
                "max_match_length": 32,
                "max_penalty": 32.0,
            },
        },
    }
    (dst_dir / "haru_config.json").write_text(
        json.dumps(haru_sidecar_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.sidecar_only:
        main_bytes = (dst_dir / "model.safetensors").stat().st_size
    else:
        main_bytes = main_b.write(dst_dir / "model.safetensors", "model.safetensors")
    if args.model_only:
        side_bytes = (dst_dir / "haru_sidecar.safetensors").stat().st_size
    else:
        side_bytes = side_b.write(dst_dir / "haru_sidecar.safetensors", "haru_sidecar.safetensors")

    if not args.sidecar_only and not args.model_only:
        all_src_keys = {k for k in src.header if k != "__metadata__"}
        missing = all_src_keys - consumed
        if missing:
            raise RuntimeError(f"Unconsumed source tensors ({len(missing)}): {sorted(missing)[:10]}")

    src.close()

    # Copy tokenizer & assets cleanly (dereferencing HF cache symlinks)
    if not args.sidecar_only and not args.model_only:
        for fname in ("tokenizer.json", "tokenizer_config.json"):
            src_file = src_dir / fname
            if src_file.exists():
                shutil.copy2(os.path.realpath(src_file), dst_dir / fname)
        src_assets = src_dir / "assets"
        if src_assets.exists():
            shutil.copytree(os.path.realpath(src_assets), dst_dir / "assets", symlinks=False, dirs_exist_ok=True)

    print(
        f"Done! model.safetensors={main_bytes / 1e9:.3f} GB, "
        f"haru_sidecar.safetensors={side_bytes / 1e9:.3f} GB, "
        f"total={(main_bytes + side_bytes) / 1e9:.3f} GB"
    )


if __name__ == "__main__":
    main()
