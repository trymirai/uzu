#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

RHT_BLOCK_SIZE = 32
UINT4_MAX = 15.0
RHT_GROUP_SIZE = 32


@dataclass(frozen=True)
class Settings:
    source_model: Path
    output_model: Path
    linear_group_size: int
    mlp_group_size: int | None
    mixer_group_size: int | None
    lm_head_mode: str
    lm_head_group_size: int


def parse_args() -> Settings:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--linear-group-size", type=int, choices=(32, 64, 128, 256, 512), default=32)
    parser.add_argument("--mlp-group-size", type=int, choices=(32, 64, 128, 256, 512))
    parser.add_argument("--mixer-group-size", type=int, choices=(32, 64, 128, 256, 512))
    parser.add_argument("--lm-head-mode", choices=("uint4", "uint2"), required=True)
    parser.add_argument("--lm-head-group-size", type=int, choices=(32, 64, 128, 256, 512), required=True)
    args = parser.parse_args()
    return Settings(
        source_model=args.source_model,
        output_model=args.output_model,
        linear_group_size=args.linear_group_size,
        mlp_group_size=args.mlp_group_size,
        mixer_group_size=args.mixer_group_size,
        lm_head_mode=args.lm_head_mode,
        lm_head_group_size=args.lm_head_group_size,
    )


def unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2).to(torch.float32)


def pack_uint4(values: torch.Tensor) -> torch.Tensor:
    assert values.dtype == torch.uint8
    assert values.shape[-1] % 2 == 0
    grouped = values.reshape(*values.shape[:-1], values.shape[-1] // 2, 2)
    return ((grouped[..., 1] << 4) | grouped[..., 0]).contiguous()


def unpack_uint2(packed: torch.Tensor) -> torch.Tensor:
    b0 = packed & 0x03
    b1 = (packed >> 2) & 0x03
    b2 = (packed >> 4) & 0x03
    b3 = (packed >> 6) & 0x03
    return torch.stack((b0, b1, b2, b3), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 4).to(torch.float32)


def pack_uint2(values: torch.Tensor) -> torch.Tensor:
    assert values.dtype == torch.uint8
    assert values.shape[-1] % 4 == 0
    grouped = values.reshape(*values.shape[:-1], values.shape[-1] // 4, 4)
    return (grouped[..., 0] | (grouped[..., 1] << 2) | (grouped[..., 2] << 4) | (grouped[..., 3] << 6)).contiguous()


def dequantize_group_uint4(
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    packed_zero_points: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    unpacked_weights = unpack_uint4(packed_weights)
    unpacked_zero_points = unpack_uint4(packed_zero_points)
    output_dim, input_dim = unpacked_weights.shape
    group_count = input_dim // group_size
    grouped_weights = unpacked_weights.view(output_dim, group_count, group_size)
    grouped_zero_points = unpacked_zero_points.view(output_dim, group_count, 1)
    grouped_scales = scales.to(torch.float32).view(output_dim, group_count, 1)
    return ((grouped_weights - grouped_zero_points) * grouped_scales).view(output_dim, input_dim)


def dequantize_mlx_embedding(
    weights: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    *,
    quantization_mode: str,
    group_size: int,
) -> torch.Tensor:
    if quantization_mode == "uint2":
        unpacked_weights = unpack_uint2(weights)
    elif quantization_mode == "uint4":
        unpacked_weights = unpack_uint4(weights)
    else:
        assert quantization_mode == "uint8"
        unpacked_weights = weights.to(torch.float32)

    output_dim, input_dim = unpacked_weights.shape
    group_count = input_dim // group_size
    grouped_weights = unpacked_weights.view(output_dim, group_count, group_size)
    grouped_scales = scales.to(torch.float32).view(output_dim, group_count, 1)
    grouped_biases = biases.to(torch.float32).view(output_dim, group_count, 1)
    return (grouped_weights * grouped_scales + grouped_biases).view(output_dim, input_dim)


def hadamard_blocks_last_dim(values: torch.Tensor, block_size: int) -> torch.Tensor:
    assert values.shape[-1] % block_size == 0
    original_shape = values.shape
    result = values.reshape(-1, values.shape[-1] // block_size, block_size)
    step = 1
    while step < block_size:
        result = result.reshape(-1, values.shape[-1] // block_size, block_size // (2 * step), 2, step)
        low = result[..., 0, :]
        high = result[..., 1, :]
        result = torch.cat((low + high, low - high), dim=-1)
        step *= 2
    return result.reshape(original_shape) / (block_size**0.5)


def unwrap_rht_group_uint4(
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    packed_zero_points: torch.Tensor,
    input_factors: torch.Tensor,
    output_factors: torch.Tensor,
) -> torch.Tensor:
    weights = dequantize_group_uint4(packed_weights, scales, packed_zero_points, RHT_GROUP_SIZE)
    weights = hadamard_blocks_last_dim(weights, RHT_BLOCK_SIZE) * input_factors.to(torch.float32)
    weights = weights.transpose(0, 1)
    weights = hadamard_blocks_last_dim(weights * output_factors.to(torch.float32), RHT_BLOCK_SIZE)
    return weights.transpose(0, 1).contiguous()


def requantize_group_uint4(weights: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert weights.ndim == 2
    output_dim, input_dim = weights.shape
    assert input_dim % group_size == 0
    group_count = input_dim // group_size
    grouped = weights.to(torch.float32).view(output_dim, group_count, group_size)
    min_vals = grouped.amin(dim=-1, keepdim=True)
    max_vals = grouped.amax(dim=-1, keepdim=True)
    scales = (max_vals - min_vals) / UINT4_MAX
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    zero_points = torch.round(-min_vals / scales).clamp(0, UINT4_MAX)
    quantized = torch.round(grouped / scales + zero_points).clamp(0, UINT4_MAX).to(torch.uint8)
    return (
        pack_uint4(quantized.view(output_dim, input_dim)),
        scales.squeeze(-1).to(torch.bfloat16).contiguous(),
        pack_uint4(zero_points.to(torch.uint8).view(output_dim, group_count)),
    )


def requantize_mlx_uint4(weights: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert weights.ndim == 2
    output_dim, input_dim = weights.shape
    assert input_dim % group_size == 0
    group_count = input_dim // group_size
    grouped = weights.to(torch.float32).view(output_dim, group_count, group_size)
    min_vals = grouped.amin(dim=-1, keepdim=True)
    max_vals = grouped.amax(dim=-1, keepdim=True)
    scales = (max_vals - min_vals) / UINT4_MAX
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = torch.round((grouped - min_vals) / scales).clamp(0, UINT4_MAX).to(torch.uint8)
    return (
        pack_uint4(quantized.view(output_dim, input_dim)),
        scales.squeeze(-1).to(torch.bfloat16).contiguous(),
        min_vals.squeeze(-1).to(torch.bfloat16).contiguous(),
    )


def requantize_mlx_uint2(weights: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert weights.ndim == 2
    output_dim, input_dim = weights.shape
    assert input_dim % group_size == 0
    group_count = input_dim // group_size
    grouped = weights.to(torch.float32).view(output_dim, group_count, group_size)
    min_vals = grouped.amin(dim=-1, keepdim=True)
    max_vals = grouped.amax(dim=-1, keepdim=True)
    scales = (max_vals - min_vals) / 3.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = torch.round((grouped - min_vals) / scales).clamp(0, 3).to(torch.uint8)
    return (
        pack_uint2(quantized.view(output_dim, input_dim)),
        scales.squeeze(-1).to(torch.bfloat16).contiguous(),
        min_vals.squeeze(-1).to(torch.bfloat16).contiguous(),
    )


def linear_group_size(settings: Settings, *, is_mlp: bool, is_mixer: bool) -> int:
    if is_mlp and settings.mlp_group_size is not None:
        return settings.mlp_group_size
    if is_mixer and settings.mixer_group_size is not None:
        return settings.mixer_group_size
    return settings.linear_group_size


def rewrite_rht_config(config: dict[str, object], settings: Settings) -> None:
    def replace(node: object, path: tuple[str, ...] = ()) -> object:
        if isinstance(node, dict):
            if node.get("type") == "RHTLinearWrapperConfig":
                inner = replace(node["inner_config"], path)
                assert isinstance(inner, dict)
                assert inner["type"] == "GroupQuantizedLinearConfig"
                inner["group_size"] = linear_group_size(
                    settings,
                    is_mlp="mlp_config" in path,
                    is_mixer="mixer_config" in path,
                )
                inner["weight_quantization_mode"] = "uint4"
                return inner
            return {key: replace(value, path + (key,)) for key, value in node.items()}
        if isinstance(node, list):
            return [replace(value, path + (str(index),)) for index, value in enumerate(node)]
        return node

    replaced = replace(config)
    assert isinstance(replaced, dict)
    config.clear()
    config.update(replaced)


def rewrite_embedding_config(config: dict[str, object], settings: Settings) -> None:
    embedding_config = config["model_config"]["model_config"]["embedding_config"]
    assert embedding_config["type"] == "MLXQuantizedTiedEmbeddingConfig"
    rewritten_embedding_config = {
        "type": "MLXQuantizedOutputUntiedEmbeddingConfig",
        "input_scale": embedding_config["input_scale"],
        "logit_soft_cap": embedding_config["logit_soft_cap"],
        "group_size": embedding_config["group_size"],
        "output_group_size": settings.lm_head_group_size,
        "embedding_quantization_mode": embedding_config["embedding_quantization_mode"],
        "activation_precision": embedding_config["activation_precision"],
        "output_quantization_mode": settings.lm_head_mode,
    }
    embedding_config.clear()
    embedding_config.update(rewritten_embedding_config)


def convert_linear(source: safe_open, weight_key: str, settings: Settings) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert weight_key.endswith(".inner_linear.weights")
    base_prefix = weight_key.removesuffix(".inner_linear.weights")
    dense = unwrap_rht_group_uint4(
        source.get_tensor(weight_key),
        source.get_tensor(f"{base_prefix}.inner_linear.scales"),
        source.get_tensor(f"{base_prefix}.inner_linear.zero_points"),
        source.get_tensor(f"{base_prefix}.input_factors"),
        source.get_tensor(f"{base_prefix}.output_factors"),
    )
    group_size = linear_group_size(
        settings,
        is_mlp=".mlp." in weight_key,
        is_mixer=".mixer." in weight_key,
    )
    return requantize_group_uint4(dense, group_size)


def convert_embedding_output(
    source: safe_open,
    settings: Settings,
    config: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embedding_config = config["model_config"]["model_config"]["embedding_config"]
    dense = dequantize_mlx_embedding(
        source.get_tensor("embedding.weights"),
        source.get_tensor("embedding.scales"),
        source.get_tensor("embedding.biases"),
        quantization_mode=embedding_config["embedding_quantization_mode"],
        group_size=embedding_config["group_size"],
    )
    if settings.lm_head_mode == "uint2":
        return requantize_mlx_uint2(dense, settings.lm_head_group_size)
    assert settings.lm_head_mode == "uint4"
    return requantize_mlx_uint4(dense, settings.lm_head_group_size)


def main() -> None:
    settings = parse_args()
    if settings.output_model.exists():
        shutil.rmtree(settings.output_model)
    shutil.copytree(settings.source_model, settings.output_model)

    with open(settings.source_model / "config.json") as f:
        config = json.load(f)

    with safe_open(settings.source_model / "model.safetensors", framework="pt") as source:
        keys = list(source.keys())
        wrapped_weight_keys = sorted(key for key in keys if key.endswith(".inner_linear.weights"))
        rewrite_rht_config(config, settings)
        rewrite_embedding_config(config, settings)

        converted_tensors: dict[str, torch.Tensor] = {}
        converted_prefixes: set[str] = set()

        for index, weight_key in enumerate(wrapped_weight_keys, start=1):
            assert weight_key.endswith(".inner_linear.weights")
            base_prefix = weight_key.removesuffix(".inner_linear.weights")
            print(f"[{index}/{len(wrapped_weight_keys)}] {base_prefix}.weights", flush=True)
            converted_weights, converted_scales, converted_zero_points = convert_linear(source, weight_key, settings)
            converted_prefixes.add(base_prefix)
            converted_tensors[f"{base_prefix}.weights"] = converted_weights
            converted_tensors[f"{base_prefix}.scales"] = converted_scales
            converted_tensors[f"{base_prefix}.zero_points"] = converted_zero_points
            bias_key = f"{base_prefix}.inner_linear.biases"
            if bias_key in keys:
                converted_tensors[f"{base_prefix}.biases"] = source.get_tensor(bias_key)

        print(f"[lm_head] {settings.lm_head_mode} gs={settings.lm_head_group_size}", flush=True)
        output_weights, output_scales, output_biases = convert_embedding_output(source, settings, config)
        converted_tensors["embedding.input_weights"] = source.get_tensor("embedding.weights")
        converted_tensors["embedding.input_scales"] = source.get_tensor("embedding.scales")
        converted_tensors["embedding.input_biases"] = source.get_tensor("embedding.biases")
        converted_tensors["embedding.output.weights"] = output_weights
        converted_tensors["embedding.output.scales"] = output_scales
        converted_tensors["embedding.output.biases"] = output_biases

        for key in keys:
            if ".inner_linear." in key or key.endswith(".input_factors") or key.endswith(".output_factors"):
                continue
            if key.startswith("embedding."):
                continue
            if key.rsplit(".", 1)[0] in converted_prefixes:
                continue
            converted_tensors[key] = source.get_tensor(key)

    with open(settings.output_model / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    save_file(converted_tensors, str(settings.output_model / "model.safetensors"))


if __name__ == "__main__":
    main()
