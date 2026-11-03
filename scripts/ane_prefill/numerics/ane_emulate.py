"""Emulate the planned ANE W8A8 FFN against the exact (fp32, dequantized-W4) MLP.

Conventions (verified by semantics_search.py, rel_rms 0.5% vs GPU):
  x_rot = H32(s_in * x); fused = s_out * H32(W_up x_rot); fused = [value | gate]
  h = silu(gate) * value; h_rot = H32(s_din * h); y = s_dout * H32(W_down h_rot)

ANE plan: output rotations folded into weights (W' = S_out H32 W), per-row int8 W';
per-token int8 activations; intermediate rotation + per-token requant on the ANE.
Optional extra Hadamard of size `extra` on the activations (folded into weight columns).
"""

import sys

import numpy as np

sys.path.insert(0, "/tmp/ane_numerics_scripts")
from mlp_ref import I_DIM, SafeTensors, block_hadamard, dequantize, hadamard, load_layer, rel_rms, silu  # noqa: E402

layers = [int(value) for value in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 8, 16, 24, 31]
count = int(sys.argv[2]) if len(sys.argv) > 2 else 128
tensors = SafeTensors("/Users/yehchenchen/models/Qwen3.5-9B-M/model.safetensors")


def fp16(values):
    return values.astype(np.float16).astype(np.float32)


def quantize_rows(matrix):
    """Symmetric int8 per row (last axis reduced). Returns dequantized values."""
    scale = np.maximum(np.abs(matrix).max(axis=-1, keepdims=True), 1e-12) / 127.0
    return np.clip(np.round(matrix / scale), -127, 127) * scale


def rows_hadamard(matrix):
    """Apply H32 across rows (output dimension) in blocks of 32: returns H32-blocked @ matrix."""
    rows, columns = matrix.shape
    blocks = matrix.reshape(rows // 32, 32, columns)
    return np.einsum("ij,bjk->bik", hadamard(32), blocks).reshape(rows, columns)


SIGNS = np.where(np.random.default_rng(20261103).random(I_DIM) < 0.5, -1.0, 1.0)


def extra_rotation(values, size):
    """Orthonormal H_size * diag(random signs) over blocks of `size` along the last axis.

    The sign flip is required: Sylvester H_size composed with the existing H32 blocks
    collapses to H_(size/32) (x) I_32 without it."""
    if size <= 1:
        return values
    shape = values.shape
    values = values * SIGNS[: shape[-1]]
    blocks = values.reshape(*shape[:-1], shape[-1] // size, size)
    return (blocks @ hadamard(size).T).reshape(shape)


def run(layer, extra_in, extra_mid, quantize_weights=True, quantize_acts=True):
    prefix = f"decoder.transformer.layers.{layer}.mlp."
    sign = {
        key: tensors.get(prefix + f"{proj}.weights.incoherence_signs.{side}_signs").astype(np.float32)
        for key, proj, side in [
            ("up_in", "up_projection", "input"),
            ("up_out", "up_projection", "output"),
            ("down_in", "down_projection", "input"),
            ("down_out", "down_projection", "output"),
        ]
    }
    up = dequantize(tensors, prefix + "up_projection.", True)
    down = dequantize(tensors, prefix + "down_projection.", True)
    x, _ = load_layer(layer)
    x = x[:count]

    # exact reference in fp32
    x_rot = block_hadamard(x * sign["up_in"])
    fused = sign["up_out"] * block_hadamard(x_rot @ up.T)
    hidden = silu(fused[:, I_DIM:]) * fused[:, :I_DIM]
    reference = sign["down_out"] * block_hadamard(block_hadamard(hidden * sign["down_in"]) @ down.T)

    # staged weights: fold output rotation, optional extra input rotation into columns, per-row int8
    up_staged = sign["up_out"][:, None] * rows_hadamard(up)
    down_staged = sign["down_out"][:, None] * rows_hadamard(down)
    up_staged = extra_rotation(up_staged, extra_in)  # rows r -> r D H (same orthonormal map as activations)
    down_staged = extra_rotation(down_staged, extra_mid)
    if quantize_weights:
        up_staged = quantize_rows(up_staged)
        down_staged = quantize_rows(down_staged)

    # GPU pack: rotate, extra rotation, per-token int8
    activations = extra_rotation(x_rot, extra_in)
    if quantize_acts:
        activations = quantize_rows(activations)
    fused_ane = fp16(activations @ up_staged.T)
    gate, value = fused_ane[:, I_DIM:], fused_ane[:, :I_DIM]
    silu_tanh = fp16(gate * 0.5 * (1.0 + np.tanh(gate * 0.5)))
    hidden_ane = fp16(silu_tanh * value)
    # ANE: intermediate rotation (grouped conv), optional extra rotation, per-token requant
    hidden_rot = fp16(extra_rotation(block_hadamard(hidden_ane * sign["down_in"]), extra_mid))
    if quantize_acts:
        hidden_rot = quantize_rows(hidden_rot)
    output = fp16(hidden_rot @ down_staged.T)
    return rel_rms(output, reference)


for layer in layers:
    weights_only = run(layer, 1, 1, quantize_acts=False)
    acts_only = run(layer, 1, 1, quantize_weights=False)
    row = [f"layer {layer:2}: W-only {weights_only:.4f}  A-only {acts_only:.4f}"]
    for extra_in, extra_mid in [(1, 1), (4096, 1), (4096, 512), (4096, 2048)]:
        row.append(f"W8A8 extra({extra_in},{extra_mid}) {run(layer, extra_in, extra_mid):.4f}")
    print("  ".join(row), flush=True)
