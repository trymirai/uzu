#!/usr/bin/env python3
"""Convert the LiquidAI LFM3-Omni-2B-MLX-4bit HF bundle into a uzu bundle.

Repack only — no re-quantization. Combines:
- Language model backbone (`model.safetensors` + `config.json`, as lfm3-audio-convert.py)
- Vision tower + multimodal projector (`vision.safetensors` + `vision_config.json`, as lfm3-convert.py)
- Audio encoder + downsampler + projector + depth decoder + LiquidCodec
  (`audio.safetensors` + `audio_config.json` + `audio_tokenizer/`, as lfm3-audio-convert.py)
"""

import argparse
import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np

SRC = Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM3-Omni-2B-MLX-4bit/snapshots"

MLX_PREFIX = "language_model.model."
UZU_LAYERS = "decoder.transformer.layers"
ENC_LAYERS = "audio.encoder.transformer.layers"
DEFAULT_GROUP = 64

NORM_CONFIG = {
    "epsilon": 1e-05,
    "scale_offset": None,
    "upcast_mode": "only_normalization",
    "subtract_mean": False,
    "has_scale": True,
    "has_biases": False,
}


class SafetensorsReader:
    """Lazy safetensors reader (raw bytes per tensor)."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
            self._data_offset = 8 + n
        self._f = open(path, "rb")

    def has(self, name: str) -> bool:
        return name in self.header

    def tensor(self, name: str) -> np.ndarray:
        info = self.header[name]
        begin, end = info["data_offsets"]
        self._f.seek(self._data_offset + begin)
        raw = self._f.read(end - begin)
        dtype = {
            "BF16": "<u2",
            "F32": "<f4",
            "U32": "<u4",
            "U8": "<u1",
            "F16": "<u2",
            "I64": "<i8",
        }[info["dtype"]]
        return np.frombuffer(raw, dtype=dtype).reshape(info["shape"])


def mlxbf16_to_f32(t: np.ndarray) -> np.ndarray:
    """MLX BF16 stored as '<u2' -> F32."""
    return (t.astype("<u4") << 16).view("<f4")


def unpack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Unpack bit-packed MLX uint32 codes [N, words] -> uint8 [N, K]."""
    n, _ = codes.shape
    k = codes.shape[1] * 32 // bits
    words = codes.astype(np.uint64)
    unpacked = np.zeros((n, k), dtype=np.uint8)
    mask = np.uint64((1 << bits) - 1)
    for idx in range(k):
        bit_off = idx * bits
        word, shift = bit_off >> 5, bit_off & 31
        val = words[:, word] >> np.uint64(shift)
        if shift + bits > 32:
            val |= words[:, word + 1] << np.uint64(32 - shift)
        unpacked[:, idx] = (val & mask).astype(np.uint8)
    return unpacked


def repack_to_u8(codes: np.ndarray, bits: int) -> np.ndarray:
    """Losslessly repack sub-8-bit MLX codes into uint8 (as uint32 words)."""
    unpacked = unpack_codes(codes, bits)
    assert unpacked.shape[1] % 4 == 0, f"expected K divisible by 4, got {unpacked.shape}"
    return np.ascontiguousarray(unpacked).view(np.uint32)


def f32_to_bf16(t: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even F32 -> BF16 stored as '<u2'."""
    u32 = t.astype("<f4").view("<u4")
    rounding_bias = 0x7FFF + ((u32 >> 16) & 1)
    return ((u32 + rounding_bias) >> 16).astype("<u2")


def write_safetensors(path: Path, blobs: dict, specs: dict):
    header = {"__metadata__": specs} if specs else {}
    offset = 0
    order = sorted(blobs.keys())
    for name in order:
        dtype, shape, data = blobs[name]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for name in order:
            f.write(blobs[name][2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the LiquidAI LFM3-Omni-2B MLX checkpoint into an uzu bundle.")
    parser.add_argument("--src", type=Path, default=SRC, help="MLX source root (default: the HF cache snapshot dir)")
    parser.add_argument("--dst", type=Path, required=True, help="output uzu bundle directory")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="pin one snapshot dir under --src (default: the newest one, by mtime)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src_root = args.src
    dst_root = args.dst

    candidates = sorted(src_root.glob("*/")) if src_root == SRC else [src_root]
    src = candidates[0] if len(candidates) == 1 else max(candidates, key=lambda p: p.stat().st_mtime)
    if args.snapshot is not None:
        src = args.snapshot
    print(f"source: {src}")

    cfg = json.load(open(src / "config.json"))
    proc_cfg = json.load(open(src / "processor_config.json"))
    gen_cfg = cfg.get("generation_config", {})

    text_cfg = cfg["text_config"]
    layer_types = text_cfg["layer_types"]
    num_layers = text_cfg["num_hidden_layers"]
    hidden = text_cfg["hidden_size"]
    inter = text_cfg["intermediate_size"]
    head_dim = text_cfg["head_dim"]
    n_heads = text_cfg["num_attention_heads"]
    n_kv = text_cfg["num_key_value_heads"]
    sliding = text_cfg["sliding_window"]
    q_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim

    quant_cfg = cfg["quantization"]
    group_size = quant_cfg["group_size"]
    default_bits = quant_cfg["bits"]
    assert group_size == DEFAULT_GROUP and default_bits in (4, 8)

    def bits_of(tensor_name: str) -> int:
        entry = quant_cfg.get(tensor_name, quant_cfg.get(tensor_name.removesuffix(".weight"), {}))
        return entry.get("bits", default_bits) if isinstance(entry, dict) else default_bits

    reader = SafetensorsReader(src / "model.safetensors")
    consumed_keys = set()

    def read_t(name: str) -> np.ndarray:
        consumed_keys.add(name)
        return reader.tensor(name)

    blobs = {}
    specs = {}

    def put(name, dtype, shape, data):
        blobs[name] = (dtype, tuple(shape), data)

    def put_spec(dst_prefix, spec):
        specs[f"{dst_prefix}.spec"] = json.dumps(spec)

    def put_quant(dst_prefix, codes, scales, biases, bits, layout, leaf="weights"):
        put(f"{dst_prefix}.{leaf}.weights", "U8", (codes.shape[0], codes.shape[1] * 4), codes.view(np.uint8).tobytes())
        if layout == "input_output":
            st, bt = scales, biases
        else:
            st = np.ascontiguousarray(scales.T)
            bt = np.ascontiguousarray(biases.T)
        put(f"{dst_prefix}.{leaf}.scales", "BF16", st.shape, st.tobytes())
        put(f"{dst_prefix}.{leaf}.biases", "BF16", bt.shape, bt.tobytes())
        put_spec(f"{dst_prefix}.{leaf}", {"type": "MLXSpec", "bits": bits, "group_size": group_size, "layout": layout})

    def carry_linear(src_prefix, dst_prefix, gate_expand=1):
        codes = read_t(f"{src_prefix}.weight")
        scales = read_t(f"{src_prefix}.scales")
        biases = read_t(f"{src_prefix}.biases")
        bits = bits_of(f"{src_prefix}.weight")
        if bits not in (4, 8):
            codes = repack_to_u8(codes, bits)
            bits = 8
        if gate_expand > 1:
            codes = np.repeat(codes, gate_expand, axis=0)
            scales = np.repeat(scales, gate_expand, axis=0)
            biases = np.repeat(biases, gate_expand, axis=0)
        assert codes.shape[0] == scales.shape[0]
        put_quant(dst_prefix, codes, scales, biases, bits, "output_input")

    def carry_norm(src_name, dst_name):
        w = read_t(src_name)
        f32 = mlxbf16_to_f32(w)
        put(dst_name, "F32", f32.shape, f32.tobytes())

    def carry_bf16(src_name, dst_name):
        t = read_t(src_name)
        put(dst_name, "BF16", t.shape, t.tobytes())

    # ---- language model ----
    emb_codes = read_t(f"{MLX_PREFIX}embed_tokens.weight")
    emb_scales = read_t(f"{MLX_PREFIX}embed_tokens.scales")
    emb_biases = read_t(f"{MLX_PREFIX}embed_tokens.biases")
    emb_bits = bits_of(f"{MLX_PREFIX}embed_tokens.weight")
    if emb_bits not in (4, 8):
        emb_codes = repack_to_u8(emb_codes, emb_bits)
        emb_bits = 8
    put_quant("decoder.embedding", emb_codes, emb_scales, emb_biases, emb_bits, "input_output", leaf="embedding")
    carry_norm(f"{MLX_PREFIX}embedding_norm.weight", "decoder.transformer.output_norm.scales")

    for i, layer_type in enumerate(layer_types):
        lsrc = f"{MLX_PREFIX}layers.{i}"
        ldst = f"{UZU_LAYERS}.{i}"
        if layer_type == "conv":
            carry_linear(f"{lsrc}.conv.in_proj", f"{ldst}.mixer.body.in_projection")
            conv = read_t(f"{lsrc}.conv.conv.weight")
            assert conv.shape[2] == 1, f"unexpected conv kernel shape {conv.shape}"
            put(
                f"{ldst}.mixer.body.conv.weights", "F32", conv.shape[:2],
                mlxbf16_to_f32(conv.reshape(conv.shape[0], conv.shape[1])).tobytes(),
            )
            carry_linear(f"{lsrc}.conv.out_proj", f"{ldst}.mixer.body.out_projection")
        else:
            part_bits = [bits_of(f"{lsrc}.self_attn.{proj}.weight") for proj in ("q_proj", "k_proj", "v_proj", "gate_proj")]
            fused_bits = default_bits if all(b == default_bits for b in part_bits) else 8
            parts = []
            for proj, expand in (("q_proj", 1), ("k_proj", 1), ("v_proj", 1), ("gate_proj", head_dim)):
                codes = read_t(f"{lsrc}.self_attn.{proj}.weight")
                scales = read_t(f"{lsrc}.self_attn.{proj}.scales")
                biases = read_t(f"{lsrc}.self_attn.{proj}.biases")
                b = bits_of(f"{lsrc}.self_attn.{proj}.weight")
                if b != fused_bits:
                    assert fused_bits == 8
                    codes = repack_to_u8(codes, b)
                if expand > 1:
                    codes = np.repeat(codes, expand, axis=0)
                    scales = np.repeat(scales, expand, axis=0)
                    biases = np.repeat(biases, expand, axis=0)
                parts.append((codes, scales, biases))
            codes = np.concatenate([p[0] for p in parts], axis=0)
            scales = np.concatenate([p[1] for p in parts], axis=0)
            biases = np.concatenate([p[2] for p in parts], axis=0)
            assert codes.shape[0] == q_dim + kv_dim + kv_dim + q_dim
            put_quant(f"{ldst}.mixer.body.qkvg_projection", codes, scales, biases, fused_bits, "output_input")
            carry_linear(f"{lsrc}.self_attn.o_proj", f"{ldst}.mixer.body.out_projection")
            carry_norm(f"{lsrc}.self_attn.q_norm.weight", f"{ldst}.mixer.body.query_norm.scales")
            carry_norm(f"{lsrc}.self_attn.k_norm.weight", f"{ldst}.mixer.body.key_norm.scales")
        w1c = read_t(f"{lsrc}.feed_forward.w1.weight")
        w3c = read_t(f"{lsrc}.feed_forward.w3.weight")
        w1s = read_t(f"{lsrc}.feed_forward.w1.scales")
        w3s = read_t(f"{lsrc}.feed_forward.w3.scales")
        w1b = read_t(f"{lsrc}.feed_forward.w1.biases")
        w3b = read_t(f"{lsrc}.feed_forward.w3.biases")
        up_codes = np.concatenate([w3c, w1c], axis=0)
        up_scales = np.concatenate([w3s, w1s], axis=0)
        up_biases = np.concatenate([w3b, w1b], axis=0)
        assert up_codes.shape[0] == 2 * inter
        put_quant(f"{ldst}.mlp.body.up_projection", up_codes, up_scales, up_biases, default_bits, "output_input")
        carry_linear(f"{lsrc}.feed_forward.w2", f"{ldst}.mlp.body.down_projection")
        carry_norm(f"{lsrc}.operator_norm.weight", f"{ldst}.mixer.pre_norm.scales")
        carry_norm(f"{lsrc}.ffn_norm.weight", f"{ldst}.mlp.pre_norm.scales")
        carry_norm(f"{lsrc}.post_operator_norm.weight", f"{ldst}.mixer.post_norm.scales")
        carry_norm(f"{lsrc}.post_ffn_norm.weight", f"{ldst}.mlp.post_norm.scales")

    # ---- vision tower + projector ----
    vcfg = cfg["vision_config"]
    V = "vision"
    carry_bf16("vision_tower.patchifier.weight", f"{V}.patchifier.weights")
    carry_norm("vision_tower.embedding_norm.weight", f"{V}.embedding_norm.scales")

    v_layer_types = vcfg["layer_types"]
    for i, lt in enumerate(v_layer_types):
        lsrc = f"vision_tower.layers.{i}"
        ldst = f"{V}.layers.{i}"
        if lt == "full_attention":
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                carry_bf16(f"{lsrc}.self_attn.{name}.weight", f"{ldst}.mixer.{name}.weights")
            carry_norm(f"{lsrc}.self_attn.q_layernorm.weight", f"{ldst}.mixer.query_norm.scales")
            carry_norm(f"{lsrc}.self_attn.k_layernorm.weight", f"{ldst}.mixer.key_norm.scales")
        else:
            carry_bf16(f"{lsrc}.conv.in_proj.weight", f"{ldst}.mixer.in_projection.weights")
            conv = read_t(f"{lsrc}.conv.spatial_conv.weight")
            assert conv.shape[3] == 1 and conv.shape[1] == conv.shape[2]
            put(
                f"{ldst}.mixer.spatial_conv.weights",
                "F32",
                conv.shape[:3],
                mlxbf16_to_f32(conv.reshape(conv.shape[0], conv.shape[1], conv.shape[2])).tobytes(),
            )
            carry_bf16(f"{lsrc}.conv.out_proj.weight", f"{ldst}.mixer.out_projection.weights")
        carry_norm(f"{lsrc}.operator_norm.weight", f"{ldst}.pre_mixer_norm.scales")
        carry_norm(f"{lsrc}.ffn_norm.weight", f"{ldst}.pre_mlp_norm.scales")
        w1 = read_t(f"{lsrc}.feed_forward.w1.weight")
        w3 = read_t(f"{lsrc}.feed_forward.w3.weight")
        w2 = read_t(f"{lsrc}.feed_forward.w2.weight")
        put(f"{ldst}.mlp.up_projection.weights", "BF16", (w3.shape[0] + w1.shape[0], w1.shape[1]), np.concatenate([w3, w1], axis=0).tobytes())
        put(f"{ldst}.mlp.down_projection.weights", "BF16", w2.shape, w2.tobytes())

    carry_norm("multi_modal_projector.layer_norm.weight", f"{V}.projector.layer_norm.scales")
    carry_norm("multi_modal_projector.layer_norm.bias", f"{V}.projector.layer_norm.biases")
    for name in ("linear_1", "linear_2"):
        carry_bf16(f"multi_modal_projector.{name}.weight", f"{V}.projector.{name}.weights")
        carry_bf16(f"multi_modal_projector.{name}.bias", f"{V}.projector.{name}.biases")

    img_proc = proc_cfg.get("image_processor", {})
    img_size = img_proc.get("size", {})
    vision_config = {
        "type": "Lfm3VlVisionConfig",
        "model_dim": vcfg["hidden_size"],
        "hidden_dim": vcfg["intermediate_size"],
        "num_layers": vcfg["num_hidden_layers"],
        "num_heads": vcfg["num_attention_heads"],
        "num_kv_heads": vcfg["num_key_value_heads"],
        "patch_size": vcfg["patch_size"],
        "temporal_patch_size": vcfg["temporal_patch_size"],
        "in_channels": vcfg["in_channels"],
        "spatial_merge_size": vcfg["spatial_merge_size"],
        "conv_kernel_size": vcfg["conv_kernel_size"],
        "norm_epsilon": vcfg["norm_eps"],
        "rope_theta": vcfg["rope_parameters"]["rope_theta"],
        "layer_types": v_layer_types,
        "projector": {
            "hidden_dim": cfg["projector_hidden_size"],
            "epsilon": cfg["projector_layer_norm_eps"],
            "has_biases": bool(cfg.get("projector_bias", True)),
            "downsample_factor": cfg["downsample_factor"],
        },
        "image": {
            "min_pixels": img_size.get("shortest_edge", 65536),
            "max_pixels": img_size.get("longest_edge", 2097152),
            "image_mean": img_proc.get("image_mean", [0.5, 0.5, 0.5]),
            "image_std": img_proc.get("image_std", [0.5, 0.5, 0.5]),
            "rescale_factor": img_proc.get("rescale_factor", 0.00392156862745098),
        },
    }

    # ---- audio sidecar ----
    enc_cfg = cfg["audio_encoder_config"]
    enc_text = enc_cfg["text_config"]
    enc_types = enc_text["layer_types"]
    enc_hidden = enc_text["hidden_size"]
    enc_heads = enc_text["num_attention_heads"]
    enc_kv = enc_text["num_key_value_heads"]
    enc_head_dim_raw = enc_hidden // enc_heads
    enc_head_dim = 128 if enc_head_dim_raw == 96 else enc_head_dim_raw
    enc_pad = enc_head_dim - enc_head_dim_raw
    enc_window = enc_cfg["sliding_window"]
    enc_conv_kernel = enc_text["conv_L_cache"]
    eps = enc_text["norm_eps"]
    theta = enc_text["rope_parameters"]["rope_theta"]
    conv_channels = enc_cfg["conv_channels"]
    num_mels = enc_cfg["num_mel_bins"]
    ff_dim = read_t("audio_tower.lfm.layers.0.feed_forward.w1.weight").shape[0]
    assert ff_dim == 1792, f"unexpected encoder ff dim {ff_dim}"

    dummy = np.zeros((2, enc_hidden), dtype="<u2")
    put("audio.encoder.embedding.embedding.weights", "BF16", dummy.shape, dummy.tobytes())
    put_spec(
        "audio.encoder.embedding.embedding",
        {"type": "FullPrecisionSpec", "layout": "input_output"},
    )
    carry_norm("audio_tower.lfm.embedding_norm.weight", "audio.encoder.transformer.output_norm.scales")

    def enc_norm(eps_scale=1.0):
        return {
            "epsilon": eps * eps_scale,
            "scale_offset": None,
            "upcast_mode": "only_normalization",
            "subtract_mean": False,
            "has_scale": True,
            "has_biases": False,
        }

    def enc_rope():
        return {
            "type": "UnscaledRoPEConfig",
            "base": theta,
            "max_sequence_length": enc_text["max_position_embeddings"],
            "head_dim": enc_head_dim,
            "rotary_dim": enc_head_dim_raw,
        }

    def enc_attention_layer():
        qk_eps_scale = enc_head_dim_raw / enc_head_dim
        return {
            "hidden_dim": ff_dim,
            "kv_source_layer_index": None,
            "rope_config": enc_rope(),
            "prologue_configs": [],
            "epilogue_configs": [],
            "mixer": {
                "conv_config": None,
                "pre_norm_config": enc_norm(),
                "body_config": {
                    "type": "AttentionConfig",
                    "qkvg_projection_config": {},
                    "out_projection_config": {},
                    "query_norm_config": enc_norm(qk_eps_scale),
                    "key_norm_config": enc_norm(qk_eps_scale),
                    "num_heads": enc_heads,
                    "num_groups": enc_kv,
                    "head_dim": enc_head_dim,
                    "is_causal": True,
                    "scale": 1.0 / math.sqrt(enc_head_dim_raw) if enc_pad else None,
                    "sliding_window_size": enc_window,
                    "logit_soft_cap": None,
                    "has_sinks": False,
                    "has_qkvg_biases": False,
                    "has_out_biases": False,
                    "normalize_values": False,
                    "is_kv_sharing": False,
                    "has_gate": False,
                },
                "post_norm_config": None,
            },
            "mlp": {
                "conv_config": None,
                "pre_norm_config": enc_norm(),
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
        }

    def enc_conv_layer():
        return {
            "hidden_dim": ff_dim,
            "kv_source_layer_index": None,
            "rope_config": None,
            "prologue_configs": [],
            "epilogue_configs": [],
            "mixer": {
                "conv_config": None,
                "pre_norm_config": enc_norm(),
                "body_config": {
                    "type": "ShortConvConfig",
                    "in_projection_config": {},
                    "conv_config": {"has_biases": False},
                    "out_projection_config": {},
                    "kernel_size": enc_conv_kernel,
                },
                "post_norm_config": None,
            },
            "mlp": {
                "conv_config": None,
                "pre_norm_config": enc_norm(),
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
        }

    for i, layer_type in enumerate(enc_types):
        lsrc = f"audio_tower.lfm.layers.{i}"
        ldst = f"{ENC_LAYERS}.{i}"
        if layer_type == "conv":
            carry_bf16(f"{lsrc}.conv.in_proj.weight", f"{ldst}.mixer.body.in_projection.weights.weights")
            put_spec(f"{ldst}.mixer.body.in_projection.weights", {"type": "FullPrecisionSpec", "layout": "output_input"})
            conv = read_t(f"{lsrc}.conv.conv.weight")
            assert conv.shape[2] == 1, f"unexpected encoder conv kernel shape {conv.shape}"
            put(
                f"{ldst}.mixer.body.conv.weights", "F32", conv.shape[:2],
                mlxbf16_to_f32(conv.reshape(conv.shape[0], conv.shape[1])).tobytes(),
            )
            carry_bf16(f"{lsrc}.conv.out_proj.weight", f"{ldst}.mixer.body.out_projection.weights.weights")
            put_spec(f"{ldst}.mixer.body.out_projection.weights", {"type": "FullPrecisionSpec", "layout": "output_input"})
        else:
            parts = []
            for proj, n_h in (("q_proj", enc_heads), ("k_proj", enc_kv), ("v_proj", enc_kv)):
                w = mlxbf16_to_f32(read_t(f"{lsrc}.self_attn.{proj}.weight"))
                if enc_pad:
                    w = np.pad(w.reshape(n_h, enc_head_dim_raw, enc_hidden), ((0, 0), (0, enc_pad), (0, 0))).reshape(n_h * enc_head_dim, enc_hidden)
                parts.append(w)
            fused = np.ascontiguousarray(np.concatenate(parts, axis=0))
            assert fused.shape[0] == (enc_heads + 2 * enc_kv) * enc_head_dim
            bf16 = f32_to_bf16(fused)
            put(f"{ldst}.mixer.body.qkvg_projection.weights.weights", "BF16", bf16.shape, bf16.tobytes())
            put_spec(f"{ldst}.mixer.body.qkvg_projection.weights", {"type": "FullPrecisionSpec", "layout": "output_input"})

            out_w = mlxbf16_to_f32(read_t(f"{lsrc}.self_attn.out_proj.weight"))
            if enc_pad:
                out_w = np.pad(out_w.reshape(enc_hidden, enc_heads, enc_head_dim_raw), ((0, 0), (0, 0), (0, enc_pad))).reshape(enc_hidden, enc_heads * enc_head_dim)
            out_bf16 = f32_to_bf16(np.ascontiguousarray(out_w))
            put(f"{ldst}.mixer.body.out_projection.weights.weights", "BF16", out_bf16.shape, out_bf16.tobytes())
            put_spec(f"{ldst}.mixer.body.out_projection.weights", {"type": "FullPrecisionSpec", "layout": "output_input"})

            for norm_src, norm_dst in (("q_layernorm", "query_norm"), ("k_layernorm", "key_norm")):
                nw = mlxbf16_to_f32(read_t(f"{lsrc}.self_attn.{norm_src}.weight"))
                if enc_pad:
                    nw = np.pad(nw * math.sqrt(enc_head_dim_raw / enc_head_dim), (0, enc_pad))
                put(f"{ldst}.mixer.body.{norm_dst}.scales", "F32", nw.shape, nw.astype("<f4").tobytes())

        w1 = mlxbf16_to_f32(read_t(f"{lsrc}.feed_forward.w1.weight"))
        w3 = mlxbf16_to_f32(read_t(f"{lsrc}.feed_forward.w3.weight"))
        w2 = mlxbf16_to_f32(read_t(f"{lsrc}.feed_forward.w2.weight"))
        up = f32_to_bf16(np.ascontiguousarray(np.concatenate([w3, w1], axis=0)))
        down = f32_to_bf16(np.ascontiguousarray(w2))
        put(f"{ldst}.mlp.body.up_projection.weights.weights", "BF16", up.shape, up.tobytes())
        put_spec(f"{ldst}.mlp.body.up_projection.weights", {"type": "FullPrecisionSpec", "layout": "output_input"})
        put(f"{ldst}.mlp.body.down_projection.weights.weights", "BF16", down.shape, down.tobytes())
        put_spec(f"{ldst}.mlp.body.down_projection.weights", {"type": "FullPrecisionSpec", "layout": "output_input"})
        carry_norm(f"{lsrc}.operator_norm.weight", f"{ldst}.mixer.pre_norm.scales")
        carry_norm(f"{lsrc}.ffn_norm.weight", f"{ldst}.mlp.pre_norm.scales")

    for name in ("conv_in", "conv_mid", "lin_mid", "conv_end", "lin_end"):
        carry_bf16(f"audio_tower.downsample.{name}.weight", f"audio.downsampler.{name}.weights")
        carry_bf16(f"audio_tower.downsample.{name}.bias", f"audio.downsampler.{name}.biases")
    carry_bf16("audio_tower.downsample.lin_out.weight", "audio.downsampler.lin_out.weights")
    carry_bf16("audio_tower.downsample.lin_out.bias", "audio.downsampler.lin_out.biases")

    carry_norm("audio_projector.layer_norm.weight", "audio.projector.layer_norm.scales")
    carry_bf16("audio_projector.layer_norm.bias", "audio.projector.layer_norm.biases")
    for name in ("linear_1", "linear_2"):
        carry_bf16(f"audio_projector.{name}.weight", f"audio.projector.{name}.weights")
        carry_bf16(f"audio_projector.{name}.bias", f"audio.projector.{name}.biases")

    carry_bf16("depth_decoder.depth_linear.weight", "audio.depth.linear.weights")
    carry_bf16("depth_decoder.depth_linear.bias", "audio.depth.linear.biases")
    carry_bf16("depth_decoder.mask_embedding", "audio.depth.mask")
    depth_cfg = cfg["depth_config"]
    depth_types = depth_cfg["layer_types"]
    for i, layer_type in enumerate(depth_types):
        assert layer_type == "full_attention", f"unexpected depth layer type {layer_type}"
        lsrc = f"depth_decoder.model.layers.{i}"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"):
            carry_bf16(f"{lsrc}.self_attn.{proj}.weight", f"audio.depth.model.layers.{i}.{proj}.weights")
        carry_norm(f"{lsrc}.self_attn.q_norm.weight", f"audio.depth.model.layers.{i}.query_norm.scales")
        carry_norm(f"{lsrc}.self_attn.k_norm.weight", f"audio.depth.model.layers.{i}.key_norm.scales")
        for name in ("w1", "w2", "w3"):
            carry_bf16(f"{lsrc}.feed_forward.{name}.weight", f"audio.depth.model.layers.{i}.{name}.weights")
        carry_norm(f"{lsrc}.operator_norm.weight", f"audio.depth.model.layers.{i}.pre_mixer_norm.scales")
        carry_norm(f"{lsrc}.ffn_norm.weight", f"audio.depth.model.layers.{i}.pre_mlp_norm.scales")

    num_codebooks = cfg["num_codebooks"]
    codebook_size = cfg["codebook_size"]
    embedding_norm = mlxbf16_to_f32(read_t("depth_decoder.model.embedding_norm.weight"))
    put("audio.depth.embedding_norm", "F32", embedding_norm.shape, embedding_norm.tobytes())
    for k in range(num_codebooks):
        carry_bf16(f"depth_decoder.depth_embeddings.{k}.embedding.weight", f"audio.depth.embeddings.{k}.embedding")
        carry_norm(f"depth_decoder.depth_embeddings.{k}.embedding_norm.weight", f"audio.depth.embeddings.{k}.norm")
    carry_bf16("codebook_embedding.weight", "audio.codebook_embedding")

    # Verify all source tensors were consumed
    all_src_keys = set(k for k in reader.header.keys() if k != "__metadata__")
    unconsumed = sorted(all_src_keys - consumed_keys)
    if unconsumed:
        raise AssertionError(f"Unconsumed source tensors ({len(unconsumed)}): {unconsumed[:10]}")

    # ---- audio_config.json ----
    feat_cfg = proc_cfg["feature_extractor"]
    eos_ids = cfg["eos_token_id"]
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    audio_config = {
        "type": "Lfm3AudioConfig",
        "audio_token_id": cfg["audio_token_id"],
        "audio_out_token_id": cfg["audio_out_token_id"],
        "audio_start_token_id": cfg["audio_start_token_id"],
        "text_end_token_id": cfg["text_end_token_id"],
        "mixed_start_token_id": cfg["mixed_start_token_id"],
        "eos_token_ids": [eos_ids[0]],
        "vocab_size": text_cfg["vocab_size"],
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
        "audio_eos_code": cfg["audio_eos_code"],
        "maskgit_schedule": cfg["maskgit_schedule"],
        "interleaved_n_text": cfg["interleaved_n_text"],
        "interleaved_n_audio": cfg["interleaved_n_audio"],
        "sample_rate": feat_cfg["sampling_rate"],
        "n_fft": feat_cfg["n_fft"],
        "hop_length": feat_cfg["hop_length"],
        "num_mel_bins": feat_cfg["feature_size"],
        "log_floor": feat_cfg["log_floor"],
        "norm_offset": feat_cfg["normalization_offset"],
        "norm_divisor": feat_cfg["normalization_divisor"],
        "encoder_stride": feat_cfg["encoder_stride"],
        "encoder_decoder_config": {
            "embedding_config": {
                "type": "TiedEmbeddingConfig",
                "input_scale": None,
                "logit_soft_cap": None,
                "logit_scale": None,
                "permute_indices": False,
            },
            "transformer_config": {
                "layer_configs": [
                    enc_attention_layer() if t != "conv" else enc_conv_layer() for t in enc_types
                ],
                "output_norm_config": {
                    "epsilon": eps,
                    "scale_offset": None,
                    "upcast_mode": "only_normalization",
                    "subtract_mean": False,
                    "has_scale": True,
                    "has_biases": False,
                },
                "model_dim": enc_hidden,
                "hidden_dim": ff_dim,
                "residual_config": {"type": "AdditiveResidualConfig"},
            },
            "vocab_size": 2,
            "ple_model_config": None,
            "embedding_norm_config": None,
        },
        "encoder_hidden": enc_hidden,
        "conv_channels": conv_channels,
        "encoder_sliding_window": enc_window,
        "depth": {
            "hidden": depth_cfg["hidden_size"],
            "layers": depth_cfg["num_hidden_layers"],
            "heads": depth_cfg["num_attention_heads"],
            "kv_heads": depth_cfg["num_key_value_heads"],
            "head_dim": depth_cfg["head_dim"],
            "intermediate": depth_cfg["intermediate_size"],
            "eps": depth_cfg["norm_eps"],
            "rope_theta": depth_cfg["rope_parameters"]["full_attention"]["rope_theta"],
        },
        "audio_sampling": {
            "temperature": gen_cfg.get("audio_temperature", 0.4),
            "top_k": gen_cfg.get("audio_top_k") or 0,
            "repetition_penalty": gen_cfg.get("audio_repetition_penalty", 1.2),
            "repetition_window": gen_cfg.get("audio_repetition_window", 64),
        },
        "text_repetition_window": 32,
    }

    # ---- config.json ----
    def norm_cfg():
        return dict(NORM_CONFIG)

    def rope_config(layer_type):
        rp = text_cfg["rope_parameters"][layer_type]
        rotary = int(head_dim * rp.get("partial_rotary_factor", 1.0))
        return {
            "type": "UnscaledRoPEConfig",
            "base": rp["rope_theta"],
            "max_sequence_length": text_cfg["max_position_embeddings"],
            "head_dim": head_dim,
            "rotary_dim": rotary,
        }

    def attention_layer(layer_type):
        return {
            "hidden_dim": inter,
            "kv_source_layer_index": None,
            "rope_config": rope_config(layer_type),
            "prologue_configs": [],
            "epilogue_configs": [],
            "mixer": {
                "conv_config": None,
                "pre_norm_config": norm_cfg(),
                "body_config": {
                    "type": "AttentionConfig",
                    "qkvg_projection_config": {},
                    "out_projection_config": {},
                    "query_norm_config": norm_cfg(),
                    "key_norm_config": norm_cfg(),
                    "num_heads": n_heads,
                    "num_groups": n_kv,
                    "head_dim": head_dim,
                    "is_causal": True,
                    "scale": None,
                    "sliding_window_size": sliding if layer_type == "sliding_attention" else None,
                    "logit_soft_cap": None,
                    "has_sinks": False,
                    "has_qkvg_biases": False,
                    "has_out_biases": False,
                    "normalize_values": False,
                    "is_kv_sharing": False,
                    "has_gate": True,
                },
                "post_norm_config": norm_cfg(),
            },
            "mlp": {
                "conv_config": None,
                "pre_norm_config": norm_cfg(),
                "body_config": {
                    "type": "DenseMLPConfig",
                    "linear_config": {},
                    "activation": {"type": "SiLU", "alpha": 1.0},
                    "has_up_biases": False,
                    "has_down_biases": False,
                    "gate_clipping": None,
                    "up_clipping": None,
                },
                "post_norm_config": norm_cfg(),
            },
        }

    def conv_layer():
        return {
            "hidden_dim": inter,
            "kv_source_layer_index": None,
            "rope_config": None,
            "prologue_configs": [],
            "epilogue_configs": [],
            "mixer": {
                "conv_config": None,
                "pre_norm_config": norm_cfg(),
                "body_config": {
                    "type": "ShortConvConfig",
                    "in_projection_config": {},
                    "conv_config": {"has_biases": False},
                    "out_projection_config": {},
                    "kernel_size": text_cfg["conv_L_cache"],
                },
                "post_norm_config": norm_cfg(),
            },
            "mlp": {
                "conv_config": None,
                "pre_norm_config": norm_cfg(),
                "body_config": {
                    "type": "DenseMLPConfig",
                    "linear_config": {},
                    "activation": {"type": "SiLU", "alpha": 1.0},
                    "has_up_biases": False,
                    "has_down_biases": False,
                    "gate_clipping": None,
                    "up_clipping": None,
                },
                "post_norm_config": norm_cfg(),
            },
        }

    prompt_template = "\n".join(
        line
        for line in (src / "chat_template.jinja").read_text().splitlines()
        if line.strip() not in ("{%- generation -%}", "{%- endgeneration -%}")
    ) + "\n"

    uzu_config = {
        "type": "LanguageModelConfig",
        "token_codec_config": {
            "type": "ChatCodecConfig",
            "prompt_template": prompt_template,
            "response_format": {
                "reasoning_pattern": None,
                "prefixes": ["\n</think>"],
                "turn_pattern": None,
                "tool_calls": {},
            },
            "system_role_name": "system",
            "user_role_name": "user",
            "assistant_role_name": "assistant",
            "eos_token": "<|im_end|>",
            "bos_token": "<|startoftext|>",
            "default_system_prompt": None,
        },
        "decoder_config": {
            "embedding_config": {
                "type": "TiedEmbeddingConfig",
                "input_scale": None,
                "logit_soft_cap": None,
                "logit_scale": None,
                "permute_indices": False,
            },
            "transformer_config": {
                "layer_configs": [
                    attention_layer(t) if t != "conv" else conv_layer() for t in layer_types
                ],
                "output_norm_config": norm_cfg(),
                "model_dim": hidden,
                "hidden_dim": inter,
                "residual_config": {"type": "AdditiveResidualConfig"},
            },
            "vocab_size": text_cfg["vocab_size"],
            "ple_model_config": None,
            "embedding_norm_config": None,
        },
        "generation_config": {
            "stop_token_ids": eos_ids,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": None,
            "banned_tokens": None,
            "repetition_penalty": None,
            "presence_penalty": None,
            "frequency_penalty": None,
            "suffix_repetition_length": 1024,
        },
    }

    dst = dst_root
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copy2(src / "tokenizer.json", dst / "tokenizer.json")
    shutil.copytree(src / "audio_tokenizer", dst / "audio_tokenizer")

    (dst / "config.json").write_text(json.dumps(uzu_config, indent=2))
    (dst / "vision_config.json").write_text(json.dumps(vision_config, indent=2))
    (dst / "audio_config.json").write_text(json.dumps(audio_config, indent=2))
    (dst / "encoding.json").write_text(json.dumps({"type": "hanashi", "name": "lfm3"}))

    text_blobs = {k: v for k, v in blobs.items() if k.startswith("decoder.")}
    text_specs = {k: v for k, v in specs.items() if k.startswith("decoder.")}
    vision_blobs = {k: v for k, v in blobs.items() if k.startswith("vision.")}
    audio_blobs = {k: v for k, v in blobs.items() if k.startswith("audio.")}
    audio_specs = {k: v for k, v in specs.items() if k.startswith("audio.")}

    write_safetensors(dst / "model.safetensors", text_blobs, text_specs)
    write_safetensors(dst / "vision.safetensors", vision_blobs, {})
    write_safetensors(dst / "audio.safetensors", audio_blobs, audio_specs)

    sizes = {"U8": 1, "BF16": 2, "F32": 4}
    total = sum(int(np.prod(s)) * sizes[d] for d, s, _ in blobs.values())
    print(
        f"wrote {len(text_blobs)} language tensors + {len(vision_blobs)} vision tensors + "
        f"{len(audio_blobs)} audio tensors ({total / 1e9:.2f} GB) -> {dst}"
    )


if __name__ == "__main__":
    main()
