from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .curve import Curve

NORM_CONFIG_FIELDS = [
    "pre_mixer_norm_config",
    "post_mixer_norm_config",
    "pre_mlp_norm_config",
    "post_mlp_norm_config",
]


@dataclass
class BlockInstance:
    block_name: str
    parameters: dict
    occurrences: int


def load_config(config_path: Path) -> dict:
    return json.loads(Path(config_path).read_text())


def enumerate_block_instances(
    config: dict,
    prefill_tokens: int,
    decode_tokens: int,
) -> list[BlockInstance]:
    transformer = config["decoder_config"]["transformer_config"]
    model_dim = transformer["model_dim"]
    hidden_dim = transformer["hidden_dim"]
    vocab_size = config["decoder_config"]["vocab_size"]
    layer_configs = transformer["layer_configs"]

    layer_norm_count = sum(
        1 for layer in layer_configs for field in NORM_CONFIG_FIELDS if layer.get(field) is not None
    )
    norm_count = layer_norm_count + 1

    projection_shapes = _matmul_projection_shapes(layer_configs, model_dim, hidden_dim)
    attention_layers = [layer for layer in layer_configs if (layer.get("mixer_config") or {}).get("type") == "AttentionConfig"]
    rope_layer_count = sum(1 for layer in layer_configs if layer.get("rope_config") is not None)
    gated_mlp_dimensions = [
        layer.get("hidden_dim") or hidden_dim
        for layer in layer_configs
        if (layer.get("mlp_config") or {}).get("activation") is not None
    ]
    average_context = prefill_tokens + decode_tokens // 2

    instances: list[BlockInstance] = []
    if prefill_tokens > 0:
        instances.append(BlockInstance("full_precision_embedding", _embedding_parameters(prefill_tokens, model_dim), 1))
        instances.append(BlockInstance("rms_norm", _norm_parameters(prefill_tokens, model_dim), norm_count))
        instances.append(BlockInstance("rope", _rope_parameters(prefill_tokens), rope_layer_count))
        for gated_dimension in gated_mlp_dimensions:
            instances.append(BlockInstance("gated_act_mul", _gated_act_mul_parameters(prefill_tokens, gated_dimension), 1))
        instances.append(BlockInstance("attention_update_kv_cache", _attention_parameters(prefill_tokens), len(attention_layers)))
        for input_dimension, output_dimension, occurrences in projection_shapes:
            instances.append(
                BlockInstance("matmul", _matmul_parameters(prefill_tokens, input_dimension, output_dimension), occurrences)
            )
        instances.append(BlockInstance("matmul", _matmul_parameters(1, model_dim, vocab_size), 1))
        instances.append(BlockInstance("unified_sampling", _sampling_parameters(vocab_size), 1))
    if decode_tokens > 0:
        instances.append(BlockInstance("full_precision_embedding", _embedding_parameters(1, model_dim), decode_tokens))
        instances.append(BlockInstance("rms_norm", _norm_parameters(1, model_dim), norm_count * decode_tokens))
        instances.append(BlockInstance("rope", _rope_parameters(1), rope_layer_count * decode_tokens))
        for gated_dimension in gated_mlp_dimensions:
            instances.append(BlockInstance("gated_act_mul", _gated_act_mul_parameters(1, gated_dimension), decode_tokens))
        instances.append(
            BlockInstance("attention_single_pass", _attention_parameters(average_context), len(attention_layers) * decode_tokens)
        )
        instances.append(
            BlockInstance("attention_update_kv_cache", _attention_parameters(average_context), len(attention_layers) * decode_tokens)
        )
        for input_dimension, output_dimension, occurrences in projection_shapes:
            instances.append(
                BlockInstance("matmul", _matmul_parameters(1, input_dimension, output_dimension), occurrences * decode_tokens)
            )
        instances.append(BlockInstance("matmul", _matmul_parameters(1, model_dim, vocab_size), decode_tokens))
        instances.append(BlockInstance("unified_sampling", _sampling_parameters(vocab_size), decode_tokens))
    return instances


def _matmul_projection_shapes(
    layer_configs: list[dict],
    model_dim: int,
    hidden_dim: int,
) -> list[tuple[int, int, int]]:
    shapes: dict[tuple[int, int], int] = {}
    for layer in layer_configs:
        mixer = layer.get("mixer_config") or {}
        if mixer.get("type") == "AttentionConfig":
            query_dimension = mixer["num_heads"] * mixer["head_dim"]
            key_value_dimension = mixer["num_groups"] * mixer["head_dim"]
            _add_shape(shapes, model_dim, query_dimension + 2 * key_value_dimension)
            _add_shape(shapes, query_dimension, model_dim)
        mlp = layer.get("mlp_config") or {}
        if mlp.get("type") is not None:
            layer_hidden_dimension = layer.get("hidden_dim") or hidden_dim
            _add_shape(shapes, model_dim, layer_hidden_dimension)
            _add_shape(shapes, model_dim, layer_hidden_dimension)
            _add_shape(shapes, layer_hidden_dimension, model_dim)
    return [(input_dimension, output_dimension, occurrences) for (input_dimension, output_dimension), occurrences in shapes.items()]


def _add_shape(
    shapes: dict[tuple[int, int], int],
    input_dimension: int,
    output_dimension: int,
) -> None:
    key = (input_dimension, output_dimension)
    shapes[key] = shapes.get(key, 0) + 1


def _norm_parameters(
    tokens: int,
    model_dim: int,
) -> dict:
    return {"tokens": tokens, "model_dim": model_dim, "data_type": "BF16", "full_layer_upcast": "false"}


def _matmul_parameters(
    tokens: int,
    input_dimension: int,
    output_dimension: int,
) -> dict:
    return {
        "tokens": tokens,
        "input_dimension": input_dimension,
        "output_dimension": output_dimension,
        "data_type": "BF16",
    }


def _embedding_parameters(
    tokens: int,
    model_dim: int,
) -> dict:
    return {"tokens": tokens, "model_dim": model_dim, "data_type": "BF16"}


def _rope_parameters(tokens: int) -> dict:
    return {"tokens": tokens, "data_type": "BF16"}


def _gated_act_mul_parameters(
    tokens: int,
    gated_dim: int,
) -> dict:
    return {"tokens": tokens, "gated_dim": gated_dim, "data_type": "BF16"}


def _attention_parameters(context_length: int) -> dict:
    return {"context_length": context_length, "data_type": "BF16"}


def _sampling_parameters(vocab_size: int) -> dict:
    return {"vocab_size": vocab_size, "data_type": "BF16"}


def predict_total(
    instances: list[BlockInstance],
    curves: dict[str, Curve],
) -> tuple[float, dict[str, float], list[str]]:
    total = 0.0
    breakdown: dict[str, float] = {}
    unmodeled: set[str] = set()
    for instance in instances:
        curve = curves.get(instance.block_name)
        if curve is None:
            unmodeled.add(instance.block_name)
            continue
        value = curve.predict(instance.parameters) * instance.occurrences
        total += value
        breakdown[instance.block_name] = breakdown.get(instance.block_name, 0.0) + value
    return total, breakdown, sorted(unmodeled)


def load_whole_model_measurement(csv_path: Path) -> dict:
    row = pd.read_csv(csv_path).iloc[0]
    return {
        "prefill_tokens": int(row["prefill_tokens"]),
        "decode_tokens": int(row["decode_tokens"]),
        "prefill_seconds": float(row["prefill_seconds"]),
        "decode_seconds": float(row["decode_seconds"]),
        "total_seconds": float(row["total_seconds"]),
    }
