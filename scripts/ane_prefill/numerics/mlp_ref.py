"""Reproduce uzu's GPU dense-MLP output from mirai W4 weights (semantics probe) and
emulate the planned ANE W8A8 pipeline. Data: /tmp/ane_numerics (dumped by
dump_mlp_activations). Usage: python3 -I mlp_ref.py <layer> [tokens]"""

import itertools
import json
import struct
import sys

import numpy as np

MODEL = "/Users/yehchenchen/models/Qwen3.5-9B-M/model.safetensors"
DUMP = "/tmp/ane_numerics"
H_DIM, I_DIM = 4096, 12288


class SafeTensors:
    def __init__(self, path):
        self.file = open(path, "rb")
        size = struct.unpack("<Q", self.file.read(8))[0]
        self.header = json.loads(self.file.read(size))
        self.base = 8 + size

    def get(self, name):
        info = self.header[name]
        start, end = info["data_offsets"]
        self.file.seek(self.base + start)
        raw = self.file.read(end - start)
        dtype = info["dtype"]
        if dtype == "BF16":
            values = bf16_to_f32(np.frombuffer(raw, dtype=np.uint16))
        else:
            values = np.frombuffer(raw, dtype={"U8": np.uint8, "I32": np.int32, "F32": np.float32}[dtype])
        return values.reshape(info["shape"])


def bf16_to_f32(values):
    return (values.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values):
    bits = values.astype(np.float32).view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) << 16).astype(np.uint32).view(np.float32)


def unpack_nibbles(packed, low_first):
    low = packed & 0xF
    high = packed >> 4
    pair = (low, high) if low_first else (high, low)
    return np.stack(pair, axis=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def dequantize(tensors, prefix, low_first):
    codes = unpack_nibbles(tensors.get(prefix + "weights.quantized.weights"), low_first).astype(np.float32)
    scales = tensors.get(prefix + "weights.quantized.scales")  # [G, N]
    zeros = unpack_nibbles(tensors.get(prefix + "weights.quantized.zero_points"), low_first).astype(np.float32)
    rows, columns = codes.shape
    groups = scales.shape[0]
    group = columns // groups
    codes = codes.reshape(rows, groups, group)
    weights = scales[:, :rows].T[:, :, None] * (codes - zeros[:, :rows].T[:, :, None])
    return weights.reshape(rows, columns)


def hadamard(size):
    matrix = np.array([[1.0]])
    while matrix.shape[0] < size:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / np.sqrt(size)


H32 = hadamard(32)


def block_hadamard(values, axis=-1):
    """Apply H32 to consecutive blocks of 32 along the last axis."""
    shape = values.shape
    blocks = values.reshape(*shape[:-1], shape[-1] // 32, 32)
    return (blocks @ H32.T).reshape(shape)


def load_layer(layer):
    tokens_in = bf16_to_f32(np.fromfile(f"{DUMP}/layer{layer:02}_in.bf16", dtype=np.uint16)).reshape(-1, H_DIM)
    tokens_out = bf16_to_f32(np.fromfile(f"{DUMP}/layer{layer:02}_out.bf16", dtype=np.uint16)).reshape(-1, H_DIM)
    return tokens_in, tokens_out


def rel_rms(estimate, reference):
    return float(np.sqrt(np.mean((estimate - reference) ** 2) / np.mean(reference**2)))


def silu(values):
    return values / (1.0 + np.exp(-values))
