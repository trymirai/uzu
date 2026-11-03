"""Find the transform conventions that reproduce uzu's GPU MLP output."""

import itertools
import sys

import numpy as np

sys.path.insert(0, "/tmp/ane_numerics_scripts")
from mlp_ref import H_DIM, I_DIM, SafeTensors, block_hadamard, dequantize, load_layer, rel_rms, silu  # noqa: E402

layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
count = 64
tensors = SafeTensors("/Users/yehchenchen/models/Qwen3.5-9B-M/model.safetensors")
prefix = f"decoder.transformer.layers.{layer}.mlp."
x, reference = load_layer(layer)
x, reference = x[:count], reference[:count]

signs = {
    name: tensors.get(prefix + f"{proj}.weights.incoherence_signs.{side}_signs").astype(np.float32)
    for name, proj, side in [
        ("up_in", "up_projection", "input"),
        ("up_out", "up_projection", "output"),
        ("down_in", "down_projection", "input"),
        ("down_out", "down_projection", "output"),
    ]
}
print({name: (value.min(), value.max()) for name, value in signs.items()})


def rotate(values, sign, sign_first):
    if sign_first:
        return block_hadamard(values * sign)
    return block_hadamard(values) * sign


def unrotate(values, sign, sign_first):
    # inverse of rotate (H is symmetric orthogonal)
    if sign_first:
        return block_hadamard(values) * sign
    return block_hadamard(values * sign)


for low_first in (True, False):
    up = dequantize(tensors, prefix + "up_projection.", low_first)
    down = dequantize(tensors, prefix + "down_projection.", low_first)
    for input_signed, sign_first, up_first in itertools.product((False, True), (True, False), (True, False)):
        if input_signed:
            x_rot = block_hadamard(x) if sign_first else None
            if x_rot is None:
                continue
        else:
            x_rot = rotate(x, signs["up_in"], sign_first)
        fused = unrotate(x_rot @ up.T, signs["up_out"], sign_first)
        first, second = fused[:, :I_DIM], fused[:, I_DIM:]
        value, gate = (first, second) if up_first else (second, first)
        hidden = silu(gate) * value
        hidden_rot = rotate(hidden, signs["down_in"], sign_first)
        output = unrotate(hidden_rot @ down.T, signs["down_out"], sign_first)
        print(
            f"low_first={low_first} input_signed={input_signed} sign_first={sign_first} up_first={up_first}: "
            f"rel_rms={rel_rms(output, reference):.4f}"
        )
