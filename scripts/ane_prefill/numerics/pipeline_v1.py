"""Exact emulation of the planned v1 GPU/ANE FFN pipeline (kernel-level algebra check).

pack (GPU):      a = H4096(D_a * x)                       -> per-token int8, scale t
stage up (GPU):  r' = H4096(D_a * D_in * H32b(r))  per row r of dequantized W_up (value|gate rows)
                 -> per-row int8, scale s
ANE:             g' = (Wg' a) * sg ; v' = (Wv' a) * sv          (fp16)
                 gate  = D_og * H32b(g') ; value = D_ov * H32b(v')   (grouped conv, block 32)
                 h = silu(gate * t) * value  (probe numerics: value carries 1/t)
                 z = D_m * H512b(D_din? no: plain) h  -> per-token int8 (scale p)
stage down:      r' = D_m * H512b(D_din * H32b(r))  per row r of dequantized W_down -> per-row int8
ANE:             y' = (Wd' z) * sd  ; out row H = p
unpack (GPU):    y = D_dout * H32b(y' * p * t)
"""

import sys

import numpy as np

sys.path.insert(0, "/tmp/ane_numerics_scripts")
from mlp_ref import H_DIM, I_DIM, SafeTensors, dequantize, load_layer, rel_rms, silu  # noqa: E402


def fwht(values, block):
    """Orthonormal Sylvester Hadamard over consecutive blocks of `block` along the last axis."""
    shape = values.shape
    out = values.reshape(-1, shape[-1] // block, block).astype(np.float64).copy()
    step = 1
    while step < block:
        out = out.reshape(out.shape[0], out.shape[1], block // (2 * step), 2, step)
        left, right = out[:, :, :, 0, :].copy(), out[:, :, :, 1, :].copy()
        out[:, :, :, 0, :] = left + right
        out[:, :, :, 1, :] = left - right
        out = out.reshape(out.shape[0], out.shape[1], block)
        step *= 2
    return (out / np.sqrt(block)).reshape(shape)


def fp16(values):
    return values.astype(np.float16).astype(np.float64)


def int8_rows(matrix):
    peak = np.maximum(np.abs(matrix).max(axis=-1, keepdims=True), 1e-12)
    scale = peak / 127.0
    return np.clip(np.round(matrix / scale), -127, 127), scale


rng = np.random.default_rng(20261103)
D_a = np.where(rng.random(H_DIM) < 0.5, -1.0, 1.0)
D_m = np.where(rng.random(512) < 0.5, -1.0, 1.0)
D_m_full = np.tile(D_m, I_DIM // 512)

layers = [int(value) for value in sys.argv[1].split(",")] if len(sys.argv) > 1 else [8, 16]
count = int(sys.argv[2]) if len(sys.argv) > 2 else 64
tensors = SafeTensors("/Users/yehchenchen/models/Qwen3.5-9B-M/model.safetensors")

for layer in layers:
    prefix = f"decoder.transformer.layers.{layer}.mlp."
    sign = {
        key: tensors.get(prefix + f"{proj}.weights.incoherence_signs.{side}_signs").astype(np.float64)
        for key, proj, side in [
            ("in", "up_projection", "input"),
            ("out", "up_projection", "output"),
            ("din", "down_projection", "input"),
            ("dout", "down_projection", "output"),
        ]
    }
    up = dequantize(tensors, prefix + "up_projection.", True).astype(np.float64)
    down = dequantize(tensors, prefix + "down_projection.", True).astype(np.float64)
    x, gpu_out = load_layer(layer)
    x = x[:count].astype(np.float64)

    # exact fp32 reference
    x_rot = fwht(x * sign["in"], 32)
    fused = sign["out"] * fwht(x_rot @ up.T, 32)
    hidden = silu(fused[:, I_DIM:]) * fused[:, :I_DIM]
    reference = sign["dout"] * fwht(fwht(hidden * sign["din"], 32) @ down.T, 32)

    # --- GPU staging ---
    up_staged = fwht(D_a * sign["in"] * fwht(up, 32), H_DIM)
    down_staged = D_m_full * fwht(sign["din"] * fwht(down, 32), 512)
    up_codes, up_scales = int8_rows(up_staged)
    down_codes, down_scales = int8_rows(down_staged)
    value_codes, gate_codes = up_codes[:I_DIM], up_codes[I_DIM:]
    value_scales, gate_scales = up_scales[:I_DIM, 0], up_scales[I_DIM:, 0]

    # --- GPU pack ---
    a = fwht(D_a * x, H_DIM)
    a_codes, t = int8_rows(a)  # t: [T,1]

    # --- ANE ---
    # MIL: dequantize(scale=1/128) on int8 operands, scales stored x128 (sg/su/sd/tx).
    a_unit = fp16(a_codes / 128)
    gate_rot = fp16(fp16(a_unit @ fp16(gate_codes / 128).T) * fp16(gate_scales * 128))
    value_rot = fp16(fp16(a_unit @ fp16(value_codes / 128).T) * fp16(value_scales * 128))
    t = fp16(t * 128)
    gate = fp16(sign["out"][I_DIM:] * fwht(gate_rot, 32))
    value = fp16(sign["out"][:I_DIM] * fwht(value_rot, 32))
    gate_t = fp16(gate * t)
    h = fp16(fp16(gate_t * 0.5 * (1 + np.tanh(gate_t * 0.5))) * value)  # = true_h / t
    z = fp16(D_m_full * fwht(h, 512))
    z_codes, p = int8_rows(z)
    p = fp16(p * 128)
    y_rot = fp16(fp16(fp16(z_codes / 128) @ fp16(down_codes / 128).T) * fp16(down_scales[:, 0] * 128))

    # --- GPU unpack ---
    output = sign["dout"] * fwht(y_rot * p * t, 32)
    print(
        f"layer {layer:2}: pipeline vs exact {rel_rms(output, reference):.4f}   "
        f"exact vs gpu {rel_rms(reference, gpu_out[:count]):.4f}   pipeline vs gpu {rel_rms(output, gpu_out[:count]):.4f}",
        flush=True,
    )
