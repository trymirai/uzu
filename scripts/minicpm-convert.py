#!/usr/bin/env python3
"""Convert the MiniCPM-V-4.7-35B-A3B-JANGH2 HF bundle into a uzu bundle.

Disk strategy: APFS-clone each shard (copy-on-write, ~zero marginal space), keep
every unchanged tensor at its original data offset, and append transformed data
(row-reordered packed weights, transposed scales, bf16->f32 casts) to a tail
blob. The new JSON header is padded with spaces to the old header length
(safetensors readers ignore trailing whitespace). If the new header outgrows the
old one, the script fails loudly rather than corrupting the shard.

Uzu naming contract (see crates/uzu-engine/src/encodable_block):
  decoder.embedding.input_embedding.{weights,scales,biases}        MLXSpec input_output
  decoder.embedding.output_embedding.{weights,scales,biases}       MLXSpec output_input (lm_head)
  decoder.transformer.output_norm.scales                 f32
  decoder.transformer.layers.L.mixer.*                   DeltaNet (in_proj fused
                                                         [qkv|z|b|a], conv, a_log,
                                                         dt_bias, norm.scales, out_proj)
  decoder.transformer.layers.L.mixer.{query,key}_norm.scales
  decoder.transformer.layers.L.mixer.qkvg_projection.*   [q|k|v|gate] rows (attention layers)
  decoder.transformer.layers.L.{pre_mixer_norm,pre_mlp_norm}.scales
  decoder.transformer.layers.L.mlp.router.weights        bf16 (full precision)
  decoder.transformer.layers.L.mlp.routed_experts.{gate,up,down}_projection.weights.quantized.{weights,scales}
                                                         Jangtq2Spec (bits/alpha/beta)
  decoder.transformer.layers.L.mlp.shared_experts.{up,down}_projection.*
  decoder.transformer.layers.L.mlp.gate.weights          shared-expert gate, bf16
"""
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SRC", "")
DST = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("DST", "")
if not SRC or not DST:
    raise SystemExit("Usage: minicpm-convert.py <SRC> <DST> (or set SRC/DST env vars)")
P = "language_model.model.layers"
UB = "decoder.transformer.layers"
LM = "language_model"
G = 64  # affine group size
MLX_AFFINE_SPEC_II = {"type": "MLXSpec", "bits": 8, "group_size": G, "layout": "input_output"}
MLX_AFFINE_SPEC_OI = {"type": "MLXSpec", "bits": 8, "group_size": G, "layout": "output_input"}
FP_SPEC_OI = {"type": "FullPrecisionSpec", "layout": "output_input"}
CODEBOOKS = {2: (0.893, 0.05065), 3: (0.47125, 0.0116)}
LAYERS = int(os.environ.get("MINICPM_LAYERS", "40"))
ATT_EVERY = 4
HIDDEN = 2048
ATT_HEADS = (16, 2, 256)  # q heads, kv heads, head dim
QKV_OUT = 8192  # in_proj_qkv output
Z_OUT = 4096
V_OUT = Z_OUT  # GDN value dim
INTER = 512  # moe intermediate
VOCAB = 248144
NUM_EXPERTS = 256


def norm_cfg():
    return {
        "epsilon": 1e-06,
        "scale_offset": 1.0,
        "upcast_mode": "only_normalization",
        "subtract_mean": False,
        "has_scale": True,
        "has_biases": False,
    }


def layer_config(i: int) -> dict:
    is_attn = (i + 1) % ATT_EVERY == 0
    qh, kvh, hd = ATT_HEADS
    mixer = {
        "type": "AttentionConfig",
        "qkvg_projection_config": {},
        "out_projection_config": {},
        "query_norm_config": norm_cfg(),
        "key_norm_config": norm_cfg(),
        "num_heads": qh,
        "num_groups": kvh,
        "head_dim": hd,
        "is_causal": True,
        "scale": None,
        "sliding_window_size": None,
        "logit_soft_cap": None,
        "has_sinks": False,
        "has_qkvg_biases": False,
        "has_out_biases": False,
        "normalize_values": False,
        "is_kv_sharing": False,
        "has_gate": True,
    } if is_attn else {
        "type": "DeltaNetConfig",
        "in_proj_config": {},
        "conv_config": {"has_biases": False},
        "out_proj_config": {},
        "norm_config": {"epsilon": 1e-06, "scale_offset": 1.0, "upcast_mode": "only_normalization", "subtract_mean": False, "has_scale": True, "has_biases": False},
        "num_heads": V_OUT // 128,  # uzu num_heads = value heads (32)
        "num_groups": (QKV_OUT - V_OUT) // 2 // 128,  # uzu num_groups = key heads (16)
        "head_dim": 128,
        "value_head_dim": 128,
        "kernel_size": 4,
    }
    rope = {
        "type": "UnscaledRoPEConfig",
        "base": 10000000.0,
        "max_sequence_length": 262144,
        "head_dim": 64,
    }
    return {
        "mixer_conv_config": None,
        "mlp_conv_config": None,
        "pre_mixer_norm_config": norm_cfg(),
        "mixer_config": mixer,
        "post_mixer_norm_config": None,
        "pre_mlp_norm_config": norm_cfg(),
        "mlp_config": {
            "type": "MixtureOfExpertsConfig",
            "expert_config": {
                "type": "DenseMLPConfig",
                "linear_config": {},
                "activation": {"type": "SiLU", "alpha": 1.0},
                "has_up_biases": False,
                "has_down_biases": False,
                "gate_clipping": None,
                "up_clipping": None,
            },
            "router_config": {},
            "routing_function": {"type": "SoftmaxRouting"},
            "num_routed_experts": NUM_EXPERTS,
            "num_active_routed_experts": 8,
            "router_has_biases": False,
            "num_shared_experts": 1,
            "expert_hidden_dim": INTER,
            "gate_config": {},
        },
        "post_mlp_norm_config": None,
        "hidden_dim": None,
        "ple_config": None,
        "has_post_layer_scalar": False,
        "kv_source_layer_index": None,
        "rope_config": rope if is_attn else None,
    }


def uzu_config(chat_template: str) -> dict:
    return {
        "type": "LanguageModelConfig",
        "decoder_config": {
            "embedding_config": {
                "type": "UntiedEmbeddingConfig",
                "input_scale": None,
                "logit_scale": None,
                "logit_soft_cap": None,
            },
            "transformer_config": {
                "layer_configs": [layer_config(i) for i in range(LAYERS)],
                "output_norm_config": norm_cfg(),
                "model_dim": HIDDEN,
                "hidden_dim": INTER,
            },
            "vocab_size": VOCAB,
            "ple_model_config": None,
            "embedding_norm_config": None,
        },
        "generation_config": {
            "stop_token_ids": [248046, 248044],
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
            "min_p": None,
            "banned_tokens": None,
            "repetition_penalty": None,
            "presence_penalty": None,
            "frequency_penalty": None,
            "suffix_repetition_length": 1024,
        },
        "token_codec_config": {
            "type": "ChatCodecConfig",
            "prompt_template": chat_template,
            "output_parser_regex": "(?s)(?=(<think>|.*?</think>)?)(?(1)(?:<think>)?(?P<chain_of_thought>.*?)(?:</think>\\s*|\\Z))(?P<response>.*)\\Z",
            "system_role_name": "system",
            "user_role_name": "user",
            "assistant_role_name": "assistant",
            "eos_token": "<|im_end|>",
            "bos_token": None,
            "default_system_prompt": None,
            "end_of_thinking_tag": "\n</think>",
            "reasoning_config": {
                "default_reasoning_effort": "no_reasoning",
                "field_name": "enable_thinking",
                "reasoning_effort_to_field_value": {"medium": True, "no_reasoning": False},
            },
        },
    }


def vision_config() -> dict:
    return {
        "vision_config": {
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_channels": 3,
            "image_size": 980,
            "patch_size": 14,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-06,
            "window_kernel_size": [2, 2],
        },
        "canvas_config": {
            "insert_layer_id": 6,
            "downsample_mode": "16x",
            "mrope_mode": "canvas",
            "merge_kernel_size": [2, 2],
            "merger_times": 1,
            "image_start_id": 248078,
            "image_end_id": 248079,
            "slice_start_id": 248088,
            "slice_end_id": 248089,
            "image_token_id": 248056,
            "newline_id": 271,
        },
    }


class Shard:
    def __init__(self, name):
        self.name = name
        self.src_path = os.path.join(SRC, name)
        self.dst_path = os.path.join(os.environ.get("MINICPM_DST", DST), name)
        with open(self.src_path, "rb") as f:
            (self.old_hlen,) = struct.unpack("<Q", f.read(8))
            self.old_header = json.loads(f.read(self.old_hlen))
        self.data_start = 8 + self.old_hlen
        self.file_size = os.path.getsize(self.src_path)
        self.new_header = {"__metadata__": {}}
        self.tail = bytearray()
        self.src_file = open(self.src_path, "rb")

    def raw(self, name):
        info = self.old_header[name]
        self.src_file.seek(self.data_start + info["data_offsets"][0])
        return self.src_file.read(info["data_offsets"][1] - info["data_offsets"][0])

    def entry(self, name):
        return self.old_header[name]

    def append(self, data: bytes, dtype: str, shape) -> dict:
        # data_offsets are relative to the data section (after the header), like the
        # source entries; the tail lives right after the original data section.
        begin = self.file_size - 8 - self.old_hlen + len(self.tail)
        self.tail += data
        return {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, begin + len(data)]}

    def bytes_u8_matrix(self, name):
        """MLX U32 [N, K/4] words -> uint8 matrix [N, K] (LSB-first codes, byte-per-code)."""
        raw = np.frombuffer(self.raw(name), dtype="<u4")
        n = self.entry(name)["shape"][0]
        m = raw.reshape(n, -1)
        return m.view(np.uint8).reshape(n, -1)  # [N, K] little-endian byte per code

    def bf16_matrix(self, name):
        raw = np.frombuffer(self.raw(name), dtype=np.uint8)
        n, g = self.entry(name)["shape"]
        return raw.reshape(n, g, 2)  # bf16 as 2-byte units


def transpose_bf16_bytes(m: np.ndarray) -> bytes:
    """[N, G, 2] bf16 bytes -> [G, N] bf16 bytes."""
    return m.transpose(1, 0, 2).tobytes()


def _layer_index(name: str):
    marker = "language_model.model.layers."
    if marker not in name:
        return None
    tail = name[len(marker):]
    digits = tail.split(".", 1)[0]
    return int(digits) if digits.isdigit() else None


def main():
    dst = os.environ.get("MINICPM_DST", DST)
    if os.path.exists(dst):
        if os.environ.get("OVERWRITE") == "1":
            print(f"overwriting existing {dst}")
            shutil.rmtree(dst)
        else:
            print(f"refusing to overwrite existing {dst}")
            sys.exit(1)
    os.makedirs(dst)
    index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    shards = {name: Shard(name) for name in sorted(set(weight_map.values()))}
    meta_shard = shards[sorted(shards.keys())[-1]]
    consumed = set()

    def set_meta(key: str, val: str):
        meta_shard.new_header["__metadata__"][key] = val

    def consume(name):
        assert name in weight_map, f"unknown tensor {name}"
        assert name not in consumed, f"double use of {name}"
        consumed.add(name)
        return shards[weight_map[name]]

    def affine_direct(old_prefix, new_prefix, spec_meta):
        """Weights retype U32->U8 at original offset; scales/biases transposed to tail."""
        sh = consume(f"{old_prefix}.weight")
        entry = dict(sh.entry(f"{old_prefix}.weight"))
        n, k4 = entry["shape"]
        assert entry["dtype"] == "U32"
        entry["dtype"] = "U8"
        entry["shape"] = [n, k4 * 4]
        sh.new_header[f"{new_prefix}.weights.weights"] = entry
        for kind in ("scales", "biases"):
            consume(f"{old_prefix}.{kind}")
            m = sh.bf16_matrix(f"{old_prefix}.{kind}")
            assert m.shape[0] == n
            g = m.shape[1]
            sh.new_header[f"{new_prefix}.weights.{kind}"] = sh.append(transpose_bf16_bytes(m), "BF16", [g, n])
        set_meta(f"{new_prefix}.weights.spec", json.dumps(spec_meta, separators=(",", ":")))

    def affine_rows_concat(slice_specs, new_prefix, total_rows):
        """Row-reordered packed weights + row-reordered scales/biases -> tail.

        slice_specs: list of (tensor_stem, row_begin, row_end); a stem may repeat with
        different ranges (qkvg splits q_proj into q and gate halves).
        """
        stems = {stem for stem, _, _ in slice_specs}
        loaded = {}
        for stem in stems:
            sh = consume(f"{stem}.weight")
            w = sh.bytes_u8_matrix(f"{stem}.weight")
            k4 = sh.entry(f"{stem}.weight")["shape"][1]
            k = k4 * 4
            g = k // G
            sh_c = consume(f"{stem}.scales")
            consume(f"{stem}.biases")
            assert sh_c is sh
            s = sh.bf16_matrix(f"{stem}.scales")
            b = sh.bf16_matrix(f"{stem}.biases")
            loaded[stem] = (sh, w, s, b, k4, k, g)
        k4s = {v[4] for v in loaded.values()}
        assert len(k4s) == 1, k4s
        k4 = k4s.pop()
        k = k4 * 4
        g = k // G
        wparts, sparts, bparts = [], [], []
        for stem, begin, end in slice_specs:
            sh, w, s, b, _, _, _ = loaded[stem]
            assert sh.entry(f"{stem}.weight")["shape"][0] >= end
            wparts.append(w[begin:end])
            sparts.append(s[begin:end])
            bparts.append(b[begin:end])
        w = np.concatenate(wparts)
        s = np.concatenate(sparts)
        b = np.concatenate(bparts)
        assert w.shape[0] == total_rows
        sh = loaded[slice_specs[0][0]][0]
        sh.new_header[f"{new_prefix}.weights.weights"] = sh.append(w.tobytes(), "U8", [total_rows, k])
        sh.new_header[f"{new_prefix}.weights.scales"] = sh.append(transpose_bf16_bytes(s), "BF16", [g, total_rows])
        sh.new_header[f"{new_prefix}.weights.biases"] = sh.append(transpose_bf16_bytes(b), "BF16", [g, total_rows])
        set_meta(f"{new_prefix}.weights.spec", json.dumps(MLX_AFFINE_SPEC_OI, separators=(",", ":")))

    def bf16_direct(old, new, dtype="BF16"):
        sh = consume(old)
        e = dict(sh.entry(old))
        assert e["dtype"] == dtype or dtype is None
        sh.new_header[new] = e

    def bf16_to_f32(old, new):
        sh = consume(old)
        raw = np.frombuffer(sh.raw(old), dtype=np.uint8)
        assert raw.shape[0] % 2 == 0
        u16 = raw.view("<u2")
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        sh.new_header[new] = sh.append(f32.tobytes(), "F32", [raw.size // 2])

    # ---------------- per-layer mapping ----------------
    for i in range(LAYERS):
        is_attn = (i + 1) % ATT_EVERY == 0
        base = f"{P}.{i}"
        ubase = f"{UB}.{i}"
        mlp = f"{base}.mlp"
        umlp = f"{ubase}.mlp"
        if is_attn:
            qh, kvh, hd = ATT_HEADS
            qdim, kvdim = qh * hd, kvh * hd
            # q_proj in the original bundle has interleaved [head_h_q, head_h_gate] pairs of width hd=256:
            # rows h*512 .. h*512+256 are query for head h,
            # rows h*512+256 .. (h+1)*512 are gate for head h.
            # Uzu attention expects contiguous [Q (all heads), K, V, Gate (all heads)].
            q_slices = [(f"{base}.self_attn.q_proj", h * 2 * hd, h * 2 * hd + hd) for h in range(qh)]
            k_slices = [(f"{base}.self_attn.k_proj", 0, kvdim)]
            v_slices = [(f"{base}.self_attn.v_proj", 0, kvdim)]
            g_slices = [(f"{base}.self_attn.q_proj", h * 2 * hd + hd, (h + 1) * 2 * hd) for h in range(qh)]
            slices = q_slices + k_slices + v_slices + g_slices
            affine_rows_concat(slices, f"{ubase}.mixer.qkvg_projection", 2 * qdim + 2 * kvdim)
            affine_direct(f"{base}.self_attn.o_proj", f"{ubase}.mixer.out_projection", MLX_AFFINE_SPEC_OI)
            bf16_to_f32(f"{base}.self_attn.q_norm.weight", f"{ubase}.mixer.query_norm.scales")
            bf16_to_f32(f"{base}.self_attn.k_norm.weight", f"{ubase}.mixer.key_norm.scales")
        else:
            slices = [
                (f"{base}.linear_attn.in_proj_qkv", 0, QKV_OUT),
                (f"{base}.linear_attn.in_proj_z", 0, Z_OUT),
                (f"{base}.linear_attn.in_proj_b", 0, V_OUT // 128),
                (f"{base}.linear_attn.in_proj_a", 0, V_OUT // 128),
            ]
            total = QKV_OUT + Z_OUT + 2 * (V_OUT // 128)
            affine_rows_concat(slices, f"{ubase}.mixer.in_proj", total)
            affine_direct(f"{base}.linear_attn.out_proj", f"{ubase}.mixer.out_proj", MLX_AFFINE_SPEC_OI)
            # conv1d.weight bf16 [conv_dim, 4, 1] -> f32 [conv_dim, 4]
            conv_name = f"{base}.linear_attn.conv1d.weight"
            consume(conv_name)
            sh = shards[weight_map[conv_name]]
            raw = np.frombuffer(sh.raw(conv_name), dtype=np.uint8)
            conv_dim = QKV_OUT  # conv covers [q 2048 | k 2048 | v 4096] = in_proj_qkv width
            assert raw.size == conv_dim * 4 * 2
            u16 = raw.view("<u2")
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            sh.new_header[f"{ubase}.mixer.conv.weights"] = sh.append(
                f32.tobytes(), "F32", [conv_dim, 4])
            bf16_to_f32(f"{base}.linear_attn.A_log", f"{ubase}.mixer.a_log")
            bf16_to_f32(f"{base}.linear_attn.dt_bias", f"{ubase}.mixer.dt_bias")
            bf16_to_f32(f"{base}.linear_attn.norm.weight", f"{ubase}.mixer.norm.scales")
        # MoE router (bf16 fp)
        consume(f"{mlp}.gate.weight")
        sh = shards[weight_map[f"{mlp}.gate.weight"]]
        e = dict(sh.entry(f"{mlp}.gate.weight"))
        assert e["dtype"] == "BF16"
        sh.new_header[f"{umlp}.router.weights.weights"] = e
        set_meta(f"{umlp}.router.weights.spec", json.dumps(FP_SPEC_OI, separators=(",", ":")))
        # routed JANGTQ banks
        for proj, bits in (("gate_proj", 2), ("up_proj", 2), ("down_proj", 3)):
            consume(f"{mlp}.switch_mlp.{proj}.tq2_packed")
            sh = shards[weight_map[f"{mlp}.switch_mlp.{proj}.tq2_packed"]]
            packed = dict(sh.entry(f"{mlp}.switch_mlp.{proj}.tq2_packed"))
            assert packed["dtype"] == "U32"
            sh.new_header[f"{umlp}.routed_experts.{proj.replace('_proj', '_projection')}.weights.quantized.weights"] = packed
            raw_name = f"{mlp}.switch_mlp.{proj}.tq2_scales"
            consume(raw_name)
            scales = dict(sh.entry(raw_name))
            assert scales["dtype"] == "F16"
            zero_routed = os.environ.get("MINICPM_ZERO_ROUTED") == "1"
            if zero_routed:
                sh.new_header[f"{umlp}.routed_experts.{proj.replace('_proj', '_projection')}.weights.quantized.scales"] = sh.append(
                    bytes(scales["shape"][0] * scales["shape"][1] * 2), "F16", scales["shape"])
                scales = None
            if scales is not None:
                sh.new_header[f"{umlp}.routed_experts.{proj.replace('_proj', '_projection')}.weights.quantized.scales"] = scales
            alpha, beta = CODEBOOKS[bits]
            set_meta(
                f"{umlp}.routed_experts.{proj.replace('_proj', '_projection')}.weights.spec",
                json.dumps({"type": "Jangtq2Spec", "bits": bits, "alpha": alpha, "beta": beta}, separators=(",", ":")),
            )
        zero_shared = os.environ.get("MINICPM_ZERO_SHARED") == "1"
        # shared expert: uzu DenseMLP fuses value|gate blocked halves in up_projection
        if zero_shared:
            zsh = shards[weight_map[f"{mlp}.shared_expert.down_proj.weight"]]
            n_up = 2 * INTER
            zsh.new_header[f"{umlp}.shared_experts.up_projection.weights.weights"] = zsh.append(
                bytes(n_up * HIDDEN), "U8", [n_up, HIDDEN])
            zsh.new_header[f"{umlp}.shared_experts.up_projection.weights.scales"] = zsh.append(
                bytes(n_up * (HIDDEN // 64) * 2), "BF16", [HIDDEN // 64, n_up])
            zsh.new_header[f"{umlp}.shared_experts.up_projection.weights.biases"] = zsh.append(
                bytes(n_up * (HIDDEN // 64) * 2), "BF16", [HIDDEN // 64, n_up])
            set_meta(f"{umlp}.shared_experts.up_projection.weights.spec", json.dumps(
                MLX_AFFINE_SPEC_OI, separators=(",", ":")))
            n_dn = HIDDEN
            zsh.new_header[f"{umlp}.shared_experts.down_projection.weights.weights"] = zsh.append(
                bytes(n_dn * INTER), "U8", [n_dn, INTER])
            zsh.new_header[f"{umlp}.shared_experts.down_projection.weights.scales"] = zsh.append(
                bytes(n_dn * (INTER // 64) * 2), "BF16", [INTER // 64, n_dn])
            zsh.new_header[f"{umlp}.shared_experts.down_projection.weights.biases"] = zsh.append(
                bytes(n_dn * (INTER // 64) * 2), "BF16", [INTER // 64, n_dn])
            set_meta(f"{umlp}.shared_experts.down_projection.weights.spec", json.dumps(
                MLX_AFFINE_SPEC_OI, separators=(",", ":")))
            consume(f"{mlp}.shared_expert.up_proj.weight"); consume(f"{mlp}.shared_expert.up_proj.scales"); consume(f"{mlp}.shared_expert.up_proj.biases")
            consume(f"{mlp}.shared_expert.gate_proj.weight"); consume(f"{mlp}.shared_expert.gate_proj.scales"); consume(f"{mlp}.shared_expert.gate_proj.biases")
            consume(f"{mlp}.shared_expert.down_proj.weight"); consume(f"{mlp}.shared_expert.down_proj.scales"); consume(f"{mlp}.shared_expert.down_proj.biases")
        else:
            affine_rows_concat(
                [
                    (f"{mlp}.shared_expert.up_proj", 0, INTER),
                    (f"{mlp}.shared_expert.gate_proj", 0, INTER),
                ],
                f"{umlp}.shared_experts.up_projection",
                2 * INTER,
            )
            affine_direct(f"{mlp}.shared_expert.down_proj", f"{umlp}.shared_experts.down_projection", MLX_AFFINE_SPEC_OI)
        # shared expert gate (bf16 fp)
        consume(f"{mlp}.shared_expert_gate.weight")
        sh = shards[weight_map[f"{mlp}.shared_expert_gate.weight"]]
        e = dict(sh.entry(f"{mlp}.shared_expert_gate.weight"))
        assert e["dtype"] == "BF16"
        sh.new_header[f"{umlp}.gate.weights.weights"] = e
        set_meta(f"{umlp}.gate.weights.spec", json.dumps(FP_SPEC_OI, separators=(",", ":")))
        # layer norms
        bf16_to_f32(f"{base}.input_layernorm.weight", f"{ubase}.pre_mixer_norm.scales")
        bf16_to_f32(f"{base}.post_attention_layernorm.weight", f"{ubase}.pre_mlp_norm.scales")

    # ---------------- global tensors ----------------
    emb_sh = consume(f"{LM}.model.embed_tokens.weight")
    n_e = emb_sh.entry(f"{LM}.model.embed_tokens.weight")["shape"][0]
    e = dict(emb_sh.entry(f"{LM}.model.embed_tokens.weight"))
    assert e["dtype"] == "U32"
    e["dtype"] = "U8"
    e["shape"] = [n_e, e["shape"][1] * 4]
    emb_sh.new_header["decoder.embedding.input_embedding.weights"] = e
    for kind in ("scales", "biases"):
        consume(f"{LM}.model.embed_tokens.{kind}")
        m = emb_sh.bf16_matrix(f"{LM}.model.embed_tokens.{kind}")
        n, g = m.shape[0], m.shape[1]
        # embedding uses Layout::InputOutput -> params OutputGroup = [N, G]: no transpose
        emb_sh.new_header[f"decoder.embedding.input_embedding.{kind}"] = emb_sh.append(m.tobytes(), "BF16", [n, g])
    set_meta("decoder.embedding.input_embedding.spec", json.dumps(
        MLX_AFFINE_SPEC_II, separators=(",", ":")))

    head_sh = consume(f"{LM}.lm_head.weight")
    e = dict(head_sh.entry(f"{LM}.lm_head.weight"))
    assert e["dtype"] == "U32"
    n = e["shape"][0]
    e["dtype"] = "U8"
    e["shape"] = [n, e["shape"][1] * 4]
    head_sh.new_header["decoder.embedding.output_embedding.weights"] = e
    for kind in ("scales", "biases"):
        consume(f"{LM}.lm_head.{kind}")
        m = head_sh.bf16_matrix(f"{LM}.lm_head.{kind}")
        n, g = m.shape[0], m.shape[1]
        head_sh.new_header[f"decoder.embedding.output_embedding.{kind}"] = head_sh.append(
            transpose_bf16_bytes(m), "BF16", [g, n])
    set_meta("decoder.embedding.output_embedding.spec", json.dumps(
        MLX_AFFINE_SPEC_OI, separators=(",", ":")))

    bf16_to_f32(f"{LM}.model.norm.weight", "decoder.transformer.output_norm.scales")

    # ---------------- map vision tensors ----------------
    vision_count = 0
    for k in list(weight_map.keys()):
        if k in consumed:
            continue
        if k.startswith("vision_tower.") or k.startswith("merger."):
            sh = consume(k)
            sh.new_header[k] = sh.entry(k)
            vision_count += 1
    print(f"mapped {vision_count} vision tensors into bundle")

    # ---------------- leftovers must be exactly dropped upper layers ----------------
    leftover = [k for k in weight_map if k not in consumed]
    max_layer = LAYERS - 1
    extra = [
        k for k in leftover
        if not (_layer_index(k) is not None and _layer_index(k) > max_layer)
    ]
    assert not extra, f"unmapped non-vision tensors: {extra[:5]} ({len(extra)} of {len(leftover)})"
    if leftover:
        print(f"dropping {len(leftover)} tensors (layers > {max_layer})")

    # ---------------- write bundles ----------------
    weight_map_out = {}
    total_size = 0
    for name, sh in shards.items():
        blob = json.dumps(sh.new_header, separators=(",", ":")).encode()
        # copy-on-write clone + in-place patch when the new header fits the old
        # header region; otherwise rebuild the shard (full data copy for it alone).
        if len(blob) <= sh.old_hlen:
            blob += b" " * (sh.old_hlen - len(blob))
            subprocess.run(["/bin/cp", "-c", sh.src_path, sh.dst_path], check=True)
            with open(sh.dst_path, "r+b") as out:
                out.write(struct.pack("<Q", len(blob)))
                out.write(blob)
                out.seek(0, os.SEEK_END)
                out.write(bytes(sh.tail))
            mode = "clone"
        else:
            print(f"{name}: header {len(blob)} > old {sh.old_hlen}, rebuilding shard")
            with open(sh.dst_path, "wb") as out, open(sh.src_path, "rb") as src:
                out.write(struct.pack("<Q", len(blob)))
                out.write(blob)
                src.seek(sh.data_start)
                shutil.copyfileobj(src, out)
                out.write(bytes(sh.tail))
            mode = "rebuilt"
        for key, entry in sh.new_header.items():
            if key == "__metadata__":
                continue
            weight_map_out[key] = name
        total_size += os.path.getsize(sh.dst_path)
        print(f"{name}: {mode}, header {len(blob)} bytes (old {sh.old_hlen}), tail {len(sh.tail)}")

    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map_out}, f, indent=1)

    chat_template = open(os.path.join(SRC, "chat_template.jinja")).read()
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(uzu_config(chat_template), f, indent=1)
    with open(os.path.join(dst, "vision_config.json"), "w") as f:
        json.dump(vision_config(), f, indent=1)
    shutil.copyfile(os.path.join(SRC, "tokenizer.json"), os.path.join(dst, "tokenizer.json"))
    print(f"wrote {len(weight_map_out)} tensors, {total_size / 2**30:.2f} GiB -> {dst}")


if __name__ == "__main__":
    main()
