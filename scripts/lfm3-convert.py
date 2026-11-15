#!/usr/bin/env python3
"""Convert the LiquidAI LFM3-1.7B-MLX-8bit HF bundle into a uzu bundle.

Repack only — no re-quantization. The MLX affine (8-bit, group 64) tensors are
carried over as-is: packed code words are reinterpreted U32->U8, scales/biases
planes are transposed [N, G] -> [G, N] for OutputInput linears (kept [N, G] for
the InputOutput embedding table), norms are cast BF16 -> F32, the ShortConv
kernel weight is cast BF16 -> F32 with its trailing unit axis squeezed.

Attention gate: the uzu kernel applies a per-channel gate (last q_dim columns of
the fused qkvg projection), while LFM3's official gate_proj is per-head
(n_heads rows). Each head's gate row is therefore repeated head_dim times —
mathematically identical after dequantization.

MLX   -> Uzu naming (see crates/uzu-engine/src/encodable_block):
  language_model.model.embed_tokens.{weight,scales,biases}
      -> decoder.embedding.embedding.{weights,scales,biases}  MLXSpec input_output
  language_model.model.embedding_norm.weight
      -> decoder.transformer.output_norm.scales               F32
  language_model.model.layers.L.conv.in_proj.{...}
      -> decoder.transformer.layers.L.mixer.in_projection.{...}
  language_model.model.layers.L.conv.conv.weight
      -> decoder.transformer.layers.L.mixer.conv.weights      F32 [hidden, kernel]
  language_model.model.layers.L.conv.out_proj.{...}
      -> decoder.transformer.layers.L.mixer.out_projection.{...}
  language_model.model.layers.L.self_attn.{q,k,v}_proj.{...} + gate_proj.{...}
      -> decoder.transformer.layers.L.mixer.qkvg_projection.{...}  [Q|K|V|GATE]
  language_model.model.layers.L.self_attn.o_proj.{...}
      -> decoder.transformer.layers.L.mixer.out_projection.{...}
  language_model.model.layers.L.self_attn.{q,k}_norm.weight
      -> decoder.transformer.layers.L.mixer.{query,key}_norm.scales  F32
  language_model.model.layers.L.feed_forward.{w1,w3}.{...}
      -> decoder.transformer.layers.L.mlp.up_projection.{...}      [GATE|UP]
  language_model.model.layers.L.feed_forward.w2.{...}
      -> decoder.transformer.layers.L.mlp.down_projection.{...}
  language_model.model.layers.L.{operator,ffn,post_operator,post_ffn}_norm.weight
      -> decoder.transformer.layers.L.{pre_mixer,pre_mlp,post_mixer,post_mlp}_norm.scales
"""

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np

SRC = Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM3-1.7B-MLX-8bit/snapshots"

MLX_PREFIX = "language_model.model."
UZU_LAYERS = "decoder.transformer.layers"
G, BITS = 64, 8

NORM_CONFIG = {
    "epsilon": 1e-05,
    "scale_offset": None,
    "upcast_mode": "only_normalization",
    "subtract_mean": False,
    "has_scale": True,
    "has_biases": False,
}


def bf16_to_f32(raw: bytes) -> np.ndarray:
    return (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")


class SafetensorsReader:
    """Lazy multi-shard safetensors reader (raw bytes per tensor)."""

    def __init__(self, directory: Path):
        self.directory = directory
        index_path = directory / "model.safetensors.index.json"
        if index_path.exists():
            self.locations = json.load(open(index_path))["weight_map"]  # tensor -> shard file
        else:
            self.locations = {t: "model.safetensors" for t in self._tensor_names(directory / "model.safetensors")}
        self._headers = {}

    def _tensor_names(self, path: Path) -> list[str]:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return list(json.loads(f.read(n)).keys())

    def _header(self, shard: str):
        if shard not in self._headers:
            f = open(self.directory / shard, "rb")
            n = struct.unpack("<Q", f.read(8))[0]
            self._headers[shard] = (f, n, json.loads(f.read(n)))
        return self._headers[shard]

    def raw(self, name: str) -> bytes:
        shard = self.locations[name]
        f, header_len, hdr = self._header(shard)
        info = hdr[name]
        begin, end = info["data_offsets"]
        f.seek(8 + header_len + begin)
        return f.read(end - begin)

    def tensor(self, name: str) -> np.ndarray:
        shard = self.locations[name]
        _, header_len, hdr = self._header(shard)
        info = hdr[name]
        dtype = {
            "BF16": "<u2",
            "F32": "<f4",
            "U32": "<u4",
            "U8": "<u1",
            "F16": "<u2",
        }[info["dtype"]]
        arr = np.frombuffer(self.raw(name), dtype=dtype)
        return arr.reshape(info["shape"])


def mlxbf16_to_f32(t: np.ndarray) -> np.ndarray:
    """MLX BF16 stored as '<u2' -> F32."""
    return (t.astype("<u4") << 16).view("<f4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the LiquidAI LFM3-1.7B MLX checkpoint into an uzu bundle.")
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
    # LFM3-VL bundles nest the text config under "text_config".
    text_cfg = cfg.get("text_config", cfg)
    layer_types = text_cfg["layer_types"]
    num_layers = text_cfg["num_hidden_layers"]
    hidden = text_cfg["hidden_size"]
    inter = text_cfg["intermediate_size"]
    head_dim = text_cfg["head_dim"]
    n_heads = text_cfg["num_attention_heads"]
    n_kv = text_cfg["num_key_value_heads"]
    sliding = text_cfg["sliding_window"]
    eps = text_cfg["norm_eps"]
    q_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    rotary_full = int(head_dim * text_cfg["rope_parameters"]["full_attention"]["partial_rotary_factor"])
    quant_cfg = cfg.get("quantization", text_cfg.get("quantization", {}))
    assert quant_cfg.get("group_size") == G and quant_cfg.get("bits") == BITS

    reader = SafetensorsReader(src)

    blobs = {}  # name -> (dtype, shape, bytes)
    specs = {}  # metadata key -> json string

    def put(name, dtype, shape, data):
        blobs[name] = (dtype, tuple(shape), data)

    def put_spec(dst_prefix, layout):
        specs[f"{dst_prefix}.weights.spec"] = json.dumps(
            {"type": "MLXSpec", "bits": BITS, "group_size": G, "layout": layout}
        )

    def carry_linear(src_prefix, dst_prefix, gate_expand=1):
        """MLX affine linear -> uzu OutputInput linear (scales/biases transposed)."""
        codes = reader.tensor(f"{src_prefix}.weight")  # U32 [N, words]
        scales = reader.tensor(f"{src_prefix}.scales")  # BF16 [N, G]
        biases = reader.tensor(f"{src_prefix}.biases")  # BF16 [N, G]
        if gate_expand > 1:
            codes = np.repeat(codes, gate_expand, axis=0)
            scales = np.repeat(scales, gate_expand, axis=0)
            biases = np.repeat(biases, gate_expand, axis=0)
        assert codes.shape[0] == scales.shape[0]
        put(f"{dst_prefix}.weights.weights", "U8", (codes.shape[0], codes.shape[1] * 4), codes.tobytes())
        st = np.ascontiguousarray(scales.T)
        bt = np.ascontiguousarray(biases.T)
        put(f"{dst_prefix}.weights.scales", "BF16", st.shape, st.tobytes())
        put(f"{dst_prefix}.weights.biases", "BF16", bt.shape, bt.tobytes())
        put_spec(dst_prefix, "output_input")

    def carry_norm(src_name, dst_name):
        w = reader.tensor(src_name)
        f32 = mlxbf16_to_f32(w)
        put(dst_name, "F32", f32.shape, f32.tobytes())

    # Embedding (tied): carry the MLX affine 8-bit codes as-is. The readout is
    # also the embedding table, so keeping it quantized halves the per-token
    # readout bandwidth (524 MB bf16 -> ~278 MB) on the decode hot path.
    emb_codes = reader.tensor(f"{MLX_PREFIX}embed_tokens.weight")  # U32 [vocab, words]
    emb_scales = reader.tensor(f"{MLX_PREFIX}embed_tokens.scales")  # BF16 [vocab, G]
    emb_biases = reader.tensor(f"{MLX_PREFIX}embed_tokens.biases")  # BF16 [vocab, G]
    put(
        "decoder.embedding.embedding.weights",
        "U8",
        (emb_codes.shape[0], emb_codes.shape[1] * 4),
        emb_codes.view(np.uint8).tobytes(),
    )
    put("decoder.embedding.embedding.scales", "BF16", emb_scales.shape, emb_scales.tobytes())
    put("decoder.embedding.embedding.biases", "BF16", emb_biases.shape, emb_biases.tobytes())
    specs["decoder.embedding.embedding.spec"] = json.dumps(
        {"type": "MLXSpec", "bits": BITS, "group_size": G, "layout": "input_output"}
    )
    carry_norm(f"{MLX_PREFIX}embedding_norm.weight", "decoder.transformer.output_norm.scales")

    for i, layer_type in enumerate(layer_types):
        lsrc = f"{MLX_PREFIX}layers.{i}"
        ldst = f"{UZU_LAYERS}.{i}"
        if layer_type == "conv":
            carry_linear(f"{lsrc}.conv.in_proj", f"{ldst}.mixer.in_projection")
            conv = reader.tensor(f"{lsrc}.conv.conv.weight")
            assert conv.shape[2] == 1
            put(f"{ldst}.mixer.conv.weights", "F32", conv.shape[:2], mlxbf16_to_f32(conv.reshape(conv.shape[0], conv.shape[1])).tobytes())
            carry_linear(f"{lsrc}.conv.out_proj", f"{ldst}.mixer.out_projection")
        else:
            # Fused [Q|K|V|GATE]: row-concat; gate rows repeated head_dim times.
            parts = []
            for proj, expand in (("q_proj", 1), ("k_proj", 1), ("v_proj", 1), ("gate_proj", head_dim)):
                codes = reader.tensor(f"{lsrc}.self_attn.{proj}.weight")
                scales = reader.tensor(f"{lsrc}.self_attn.{proj}.scales")
                biases = reader.tensor(f"{lsrc}.self_attn.{proj}.biases")
                if expand > 1:
                    codes = np.repeat(codes, expand, axis=0)
                    scales = np.repeat(scales, expand, axis=0)
                    biases = np.repeat(biases, expand, axis=0)
                parts.append((codes, scales, biases))
            codes = np.concatenate([p[0] for p in parts], axis=0)
            scales = np.concatenate([p[1] for p in parts], axis=0)
            biases = np.concatenate([p[2] for p in parts], axis=0)
            assert codes.shape[0] == q_dim + kv_dim + kv_dim + q_dim
            put(f"{ldst}.mixer.qkvg_projection.weights.weights", "U8", (codes.shape[0], codes.shape[1] * 4), codes.tobytes())
            st = np.ascontiguousarray(scales.T)
            bt = np.ascontiguousarray(biases.T)
            put(f"{ldst}.mixer.qkvg_projection.weights.scales", "BF16", st.shape, st.tobytes())
            put(f"{ldst}.mixer.qkvg_projection.weights.biases", "BF16", bt.shape, bt.tobytes())
            put_spec(f"{ldst}.mixer.qkvg_projection", "output_input")
            carry_linear(f"{lsrc}.self_attn.o_proj", f"{ldst}.mixer.out_projection")
            carry_norm(f"{lsrc}.self_attn.q_norm.weight", f"{ldst}.mixer.query_norm.scales")
            carry_norm(f"{lsrc}.self_attn.k_norm.weight", f"{ldst}.mixer.key_norm.scales")
        # Dense MLP: up_projection = [w1 (gate) | w3 (up)], down_projection = w2.
        w1c = reader.tensor(f"{lsrc}.feed_forward.w1.weight")
        w3c = reader.tensor(f"{lsrc}.feed_forward.w3.weight")
        w1s = reader.tensor(f"{lsrc}.feed_forward.w1.scales")
        w3s = reader.tensor(f"{lsrc}.feed_forward.w3.scales")
        w1b = reader.tensor(f"{lsrc}.feed_forward.w1.biases")
        w3b = reader.tensor(f"{lsrc}.feed_forward.w3.biases")
        # Fused gate|up layout: uzu's gated-act kernel applies SiLU to the
        # FIRST half. MLX applies SiLU to w1 (gate) — but empirically uzu wants
        # [w3 | w1] here (gate = second half); see layer-0 parity debugging.
        up_codes = np.concatenate([w3c, w1c], axis=0)
        up_scales = np.ascontiguousarray(np.concatenate([w3s, w1s], axis=0).T)
        up_biases = np.ascontiguousarray(np.concatenate([w3b, w1b], axis=0).T)
        assert up_codes.shape[0] == 2 * inter
        put(f"{ldst}.mlp.up_projection.weights.weights", "U8", (up_codes.shape[0], up_codes.shape[1] * 4), up_codes.tobytes())
        put(f"{ldst}.mlp.up_projection.weights.scales", "BF16", up_scales.shape, up_scales.tobytes())
        put(f"{ldst}.mlp.up_projection.weights.biases", "BF16", up_biases.shape, up_biases.tobytes())
        put_spec(f"{ldst}.mlp.up_projection", "output_input")
        carry_linear(f"{lsrc}.feed_forward.w2", f"{ldst}.mlp.down_projection")
        carry_norm(f"{lsrc}.operator_norm.weight", f"{ldst}.pre_mixer_norm.scales")
        carry_norm(f"{lsrc}.ffn_norm.weight", f"{ldst}.pre_mlp_norm.scales")
        carry_norm(f"{lsrc}.post_operator_norm.weight", f"{ldst}.post_mixer_norm.scales")
        carry_norm(f"{lsrc}.post_ffn_norm.weight", f"{ldst}.post_mlp_norm.scales")

    # ---- vision tower + projector (LFM3-VL bundles only) ----
    is_vl = "vision_tower.patchifier.weight" in reader.locations
    vision_cfg_out = None
    if is_vl:
        vcfg = cfg.get("vision_config", {})
        V = "vision"

        def carry_bf16_raw(src_name, dst_name):
            """BF16 MLX tensor -> BF16 uzu tensor, bits preserved."""
            t = reader.tensor(src_name)
            put(dst_name, "BF16", t.shape, t.tobytes())

        def carry_norm_bf16(src_name, dst_name):
            w = reader.tensor(src_name)
            f32 = mlxbf16_to_f32(w)
            put(dst_name, "F32", f32.shape, f32.tobytes())

        carry_bf16_raw(f"vision_tower.patchifier.weight", f"{V}.patchifier.weights")
        carry_norm_bf16(f"vision_tower.embedding_norm.weight", f"{V}.embedding_norm.scales")

        v_layer_types = vcfg["layer_types"]
        for i, lt in enumerate(v_layer_types):
            lsrc = f"vision_tower.layers.{i}"
            ldst = f"{V}.layers.{i}"
            if lt == "full_attention":
                for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    carry_bf16_raw(f"{lsrc}.self_attn.{name}.weight", f"{ldst}.mixer.{name}.weights")
                carry_norm_bf16(f"{lsrc}.self_attn.q_layernorm.weight", f"{ldst}.mixer.query_norm.scales")
                carry_norm_bf16(f"{lsrc}.self_attn.k_layernorm.weight", f"{ldst}.mixer.key_norm.scales")
            else:
                carry_bf16_raw(f"{lsrc}.conv.in_proj.weight", f"{ldst}.mixer.in_projection.weights")
                conv = reader.tensor(f"{lsrc}.conv.spatial_conv.weight")
                assert conv.shape[3] == 1 and conv.shape[1] == conv.shape[2]
                put(f"{ldst}.mixer.spatial_conv.weights", "F32", conv.shape[:3], mlxbf16_to_f32(conv.reshape(conv.shape[0], conv.shape[1], conv.shape[2])).tobytes())
                carry_bf16_raw(f"{lsrc}.conv.out_proj.weight", f"{ldst}.mixer.out_projection.weights")
            carry_norm_bf16(f"{lsrc}.operator_norm.weight", f"{ldst}.pre_mixer_norm.scales")
            carry_norm_bf16(f"{lsrc}.ffn_norm.weight", f"{ldst}.pre_mlp_norm.scales")
            w1 = reader.tensor(f"{lsrc}.feed_forward.w1.weight")
            w3 = reader.tensor(f"{lsrc}.feed_forward.w3.weight")
            w2 = reader.tensor(f"{lsrc}.feed_forward.w2.weight")
            # Same fused [up | gate] convention as the language DenseMLP.
            put(f"{ldst}.mlp.up_projection.weights", "BF16", (w3.shape[0] + w1.shape[0], w1.shape[1]), np.concatenate([w3, w1], axis=0).tobytes())
            put(f"{ldst}.mlp.down_projection.weights", "BF16", w2.shape, w2.tobytes())

        # Projector: layer norm -> reshape(-1, v_hidden * merge^2) -> linear_1 -> gelu -> linear_2.
        carry_norm_bf16(f"multi_modal_projector.layer_norm.weight", f"{V}.projector.layer_norm.scales")
        ln_bias = reader.tensor(f"multi_modal_projector.layer_norm.bias")
        put(f"{V}.projector.layer_norm.biases", "F32", ln_bias.shape, mlxbf16_to_f32(ln_bias).tobytes())
        for name in ("linear_1", "linear_2"):
            carry_bf16_raw(f"multi_modal_projector.{name}.weight", f"{V}.projector.{name}.weights")
            lb = reader.tensor(f"multi_modal_projector.{name}.bias")
            put(f"{V}.projector.{name}.biases", "BF16", lb.shape, lb.tobytes())

        vision_cfg_out = {
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
                "hidden_dim": cfg.get("projector_hidden_size"),
                "epsilon": cfg.get("projector_layer_norm_eps"),
                "has_biases": bool(cfg.get("projector_bias", True)),
                "downsample_factor": cfg.get("downsample_factor"),
            },
            "image": {
                "min_pixels": 65536,
                "max_pixels": 16777216,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "rescale_factor": 0.00392156862745098,
            },
        }

    # ---- config.json ----
    def norm_cfg():
        return dict(NORM_CONFIG)

    def rope_config(layer_type):
        if layer_type == "full_attention":
            return {
                "type": "UnscaledRoPEConfig",
                "base": text_cfg["rope_parameters"]["full_attention"]["rope_theta"],
                "max_sequence_length": text_cfg["max_position_embeddings"],
                "head_dim": head_dim,
                "rotary_dim": rotary_full,
            }
        return {
            "type": "UnscaledRoPEConfig",
            "base": text_cfg["rope_parameters"]["sliding_attention"]["rope_theta"],
            "max_sequence_length": text_cfg["max_position_embeddings"],
            "head_dim": head_dim,
        }

    def attention_layer(layer_type):
        return {
            "mixer_conv_config": None,
            "mlp_conv_config": None,
            "pre_mixer_norm_config": norm_cfg(),
            "mixer_config": {
                "type": "AttentionConfig",
                "qkvg_projection_config": {},
                "out_projection_config": {},
                "query_norm_config": norm_cfg(),
                "key_norm_config": norm_cfg(),
                "num_heads": n_heads,
                "num_groups": n_heads // n_kv,
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
            "post_mixer_norm_config": norm_cfg(),
            "pre_mlp_norm_config": norm_cfg(),
            "mlp_config": {
                "type": "DenseMLPConfig",
                "linear_config": {},
                "activation": {"type": "SiLU", "alpha": 1.0},
                "has_up_biases": False,
                "has_down_biases": False,
                "gate_clipping": None,
                "up_clipping": None,
            },
            "post_mlp_norm_config": norm_cfg(),
            "hidden_dim": None,
            "ple_config": None,
            "has_post_layer_scalar": False,
            "kv_source_layer_index": None,
            "rope_config": rope_config(layer_type),
        }

    def conv_layer():
        return {
            "mixer_conv_config": None,
            "mlp_conv_config": None,
            "pre_mixer_norm_config": norm_cfg(),
            "mixer_config": {
                "type": "ShortConvConfig",
                "in_projection_config": {},
                "conv_config": {"has_biases": False},
                "out_projection_config": {},
                "kernel_size": text_cfg["conv_L_cache"],
            },
            "post_mixer_norm_config": norm_cfg(),
            "pre_mlp_norm_config": norm_cfg(),
            "mlp_config": {
                "type": "DenseMLPConfig",
                "linear_config": {},
                "activation": {"type": "SiLU", "alpha": 1.0},
                "has_up_biases": False,
                "has_down_biases": False,
                "gate_clipping": None,
                "up_clipping": None,
            },
            "post_mlp_norm_config": norm_cfg(),
            "hidden_dim": None,
            "ple_config": None,
            "has_post_layer_scalar": False,
            "kv_source_layer_index": None,
            "rope_config": None,
        }

    prompt_template = "\n".join(
        line
        for line in (src / "chat_template.jinja").read_text().splitlines()
        if line.strip() not in ("{%- generation -%}", "{%- endgeneration -%}")
    ) + "\n"

    eos_id = text_cfg["eos_token_id"]
    eos = eos_id[0] if isinstance(eos_id, list) else eos_id
    uzu_config = {
        "type": "LanguageModelConfig",
        "token_codec_config": {
            "type": "ChatCodecConfig",
            "prompt_template": prompt_template,
            "output_parser_regex": None,
            "system_role_name": "system",
            "user_role_name": "user",
            "assistant_role_name": "assistant",
            "eos_token": "<|im_end|>",
            "bos_token": "<|startoftext|>",
            "end_of_thinking_tag": "\n</think>",
            "default_system_prompt": None,
        },
        "decoder_config": {
            "embedding_config": {
                "type": "TiedEmbeddingConfig",
                "input_scale": None,
                "logit_soft_cap": None,
                "logit_scale": None,
            },
            "transformer_config": {
                "layer_configs": [
                    attention_layer(t) if t != "conv" else conv_layer() for t in layer_types
                ],
                "output_norm_config": norm_cfg(),
                "model_dim": hidden,
                "hidden_dim": inter,
            },
            "vocab_size": text_cfg["vocab_size"],
            "ple_model_config": None,
            "embedding_norm_config": None,
        },
        "generation_config": {
            "stop_token_ids": [eos],
            "temperature": 0.1,
            "top_k": 50,
            "top_p": 1.0,
            "min_p": None,
            "banned_tokens": None,
            "repetition_penalty": 1.1,
            "presence_penalty": None,
            "frequency_penalty": None,
            "suffix_repetition_length": 1024,
        },
    }

    # ---- write bundle ----
    dst = dst_root
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copy2(src / "tokenizer.json", dst / "tokenizer.json")

    (dst / "config.json").write_text(json.dumps(uzu_config, indent=2))
    (dst / "encoding.json").write_text(json.dumps({"type": "hanashi", "name": "lfm3"}))
    if vision_cfg_out is not None:
        (dst / "vision_config.json").write_text(json.dumps(vision_cfg_out, indent=2))

    # Vision tensors go to a separate safetensors file: the language model
    # loader asserts every tensor in model.safetensors is consumed, and the
    # vision tower (Phase B) will read vision.safetensors directly.
    text_blobs = {k: v for k, v in blobs.items() if not k.startswith("vision.")}
    text_specs = {k: v for k, v in specs.items() if not k.startswith("vision.")}
    vis_blobs = {k: v for k, v in blobs.items() if k.startswith("vision.")}
    write_safetensors(dst / "model.safetensors", text_blobs, text_specs)
    if vis_blobs:
        write_safetensors(dst / "vision.safetensors", vis_blobs, {})
    sizes = {"U8": 1, "BF16": 2, "F32": 4}
    total = sum(int(np.prod(s)) * sizes[d] for d, s, _ in blobs.values())
    print(f"wrote {len(text_blobs)} language tensors + {len(vis_blobs)} vision tensors "
          f"({total / 1e9:.2f} GB), specs: {len(text_specs)} -> {dst}")


def write_safetensors(path: Path, blobs: dict, specs: dict):
    header = {"__metadata__": specs}
    offset = 0
    for name, (dtype, shape, data) in blobs.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    header_json = json.dumps(header, separators=(",", ":")).encode()
    padding = (8 - (8 + len(header_json)) % 8) % 8
    header_json += b" " * padding
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for name, (_, _, data) in blobs.items():
            f.write(data)


if __name__ == "__main__":
    main()
