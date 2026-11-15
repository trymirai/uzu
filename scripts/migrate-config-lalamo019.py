#!/usr/bin/env python3
"""Migrate local model bundle config.json to the lalamo 0.19 schema (upstream 7568c26b).

Old flat TransformerLayerConfig -> nested sublayers:
  mixer_conv_config        -> mixer.conv_config
  pre_mixer_norm_config    -> mixer.pre_norm_config
  mixer_config             -> mixer.body_config
  post_mixer_norm_config   -> mixer.post_norm_config
  mlp_conv_config          -> mlp.conv_config
  pre_mlp_norm_config      -> mlp.pre_norm_config
  mlp_config               -> mlp.body_config
  post_mlp_norm_config     -> mlp.post_norm_config
  ple_config               -> epilogue_configs += PerLayerEmbeddingInjectionConfig
  has_post_layer_scalar    -> epilogue_configs += ResidualScaleConfig
New required fields: prologue_configs: [], epilogue_configs: [].

Usage: python3 scripts/migrate-config-lalamo019.py <bundle-dir> [more-dirs...]
Writes config.json next to a config.json.pre-lalamo019 backup; skips already-migrated files.
"""

import json
import shutil
import sys
from pathlib import Path

PLE_INJECTION_KEYS = ["linear_config", "norm_config", "ple_dim", "activation"]


def migrate_layer(layer: dict) -> dict:
    if "mixer" in layer:  # already migrated
        return layer
    out = dict(layer)
    mixer = {}
    mlp = {}
    for old, new in [
        ("mixer_conv_config", "conv_config"),
        ("pre_mixer_norm_config", "pre_norm_config"),
        ("mixer_config", "body_config"),
        ("post_mixer_norm_config", "post_norm_config"),
    ]:
        if old in layer:
            mixer[new] = layer[old]
    for old, new in [
        ("mlp_conv_config", "conv_config"),
        ("pre_mlp_norm_config", "pre_norm_config"),
        ("mlp_config", "body_config"),
        ("post_mlp_norm_config", "post_norm_config"),
    ]:
        if old in layer:
            mlp[new] = layer[old]
    out["mixer"] = mixer
    out["mlp"] = mlp

    epilogue = []
    if layer.get("ple_config") is not None:
        ple = {key: layer["ple_config"][key] for key in PLE_INJECTION_KEYS if key in layer["ple_config"]}
        epilogue.append({"type": "PerLayerEmbeddingInjectionConfig", **ple})
    if layer.get("has_post_layer_scalar"):
        epilogue.append({"type": "ResidualScaleConfig"})
    out["epilogue_configs"] = epilogue
    out["prologue_configs"] = []

    # lalamo 0.19 added a required output gate to DeltaNet; the engine only
    # supports SiLU there (delta_net.rs rejects anything else).
    body = out.get("mixer", {}).get("body_config")
    if isinstance(body, dict) and body.get("type") == "DeltaNetConfig" and "output_gate" not in body:
        body["output_gate"] = {"type": "SiLU", "alpha": 1.0}

    for key in [
        "mixer_conv_config", "pre_mixer_norm_config", "mixer_config", "post_mixer_norm_config",
        "mlp_conv_config", "pre_mlp_norm_config", "mlp_config", "post_mlp_norm_config",
        "ple_config", "has_post_layer_scalar",
    ]:
        out.pop(key, None)
    return out


def migrate_bundle(bundle: Path) -> None:
    config_path = bundle / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("type") != "LanguageModelConfig":
        print(f"{bundle}: not a LanguageModelConfig, skipping")
        return
    transformer = config.get("decoder_config", {}).get("transformer_config", {})
    layers = transformer.get("layer_configs", [])
    if layers and "mixer" in layers[0]:
        print(f"{bundle}: already migrated")
        return
    migrated = {**config,
                "decoder_config": {**config.get("decoder_config", {}),
                                   "transformer_config": {**transformer,
                                                          "layer_configs": [migrate_layer(l) for l in layers]}}}
    # lalamo 0.19 strict schema: newly required fields must be present.
    embedding = migrated["decoder_config"].get("embedding_config")
    if isinstance(embedding, dict) and "permute_indices" not in embedding:
        embedding["permute_indices"] = False
    if "residual_config" not in migrated["decoder_config"].get("transformer_config", {}):
        migrated["decoder_config"]["transformer_config"]["residual_config"] = {"type": "AdditiveResidualConfig"}

    # ChatCodecConfig: output_parser_regex/end_of_thinking_tag replaced by response_format.
    codec = migrated.get("token_codec_config")
    if isinstance(codec, dict) and codec.get("type") == "ChatCodecConfig" and "response_format" not in codec:
        eot = codec.pop("end_of_thinking_tag", None)
        codec.pop("output_parser_regex", None)
        codec["response_format"] = None if eot is None else {
            "reasoning_pattern": None,
            "prefixes": [eot],
            "turn_pattern": None,
            "tool_calls": {},
        }
    backup = config_path.with_name("config.json.pre-lalamo019")
    if not backup.exists():
        shutil.copy2(config_path, backup)
    config_path.write_text(json.dumps(migrated, indent=1, ensure_ascii=False) + "\n")
    print(f"{bundle}: migrated ({len(layers)} layers), backup at {backup.name}")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        migrate_bundle(Path(arg).expanduser())


if __name__ == "__main__":
    main()
