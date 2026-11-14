#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_models import fetch, write_catalog

HANASHI = Path(__file__).resolve().parents[3] / "crates/legacy/hanashi/configs"
LEVELS = ("low", "medium", "high", "xhigh")


def reasoning_efforts(encodings):
    """Maps every effort to one the model's Hanashi encoding renders, as close in meaning as the model allows."""
    mappings = []
    default = "default"
    for encoding in encodings:
        if encoding["type"] != "hanashi":
            raise ValueError(f"Unknown encoding type: {encoding['type']}")
        names = {config["name"]: config["rendering"] for config in json.loads((HANASHI / "configs.json").read_text())}
        config = json.loads((HANASHI / "rendering" / f"{names[encoding['name']]}.json").read_text())
        default = config.get("default_reasoning_effort", default)
        mappings += [
            field["mapping"]
            for role in config["rendering"].values()
            for field in (role.get("message", {}) | role.get("context", {})).values()
            if field.get("block") == "reasoning_effort"
        ]
    supported = set.intersection(*map(set, mappings)) if mappings else set()
    levels = [level for level in LEVELS if level in supported]
    # As RenderingConfig::default_reasoning_effort: the one effort that renders exactly as the default does.
    if default == "default" and any(mapping.get("default") is not None for mapping in mappings):
        same = [
            effort
            for effort in supported - {"default"}
            if all(mapping[effort] == mapping["default"] for mapping in mappings)
        ]
        if len(same) == 1:
            (default,) = same

    def resolve(effort):
        if effort == "default":
            return default
        if effort in supported:
            return effort
        if effort == "disabled":
            # A model that can't stop reasoning reasons the least it can; one that never reasons has nothing to stop.
            return levels[0] if levels else default
        # The next level up, else the highest; a model without levels reasons as it does by default.
        higher = [level for level in levels if LEVELS.index(level) > LEVELS.index(effort)]
        return (higher or levels or [default])[0 if higher else -1]

    return {effort: resolve(effort) for effort in ("disabled", "default", *LEVELS)}


def update():
    directory = Path(__file__).resolve().parent
    version_file = directory.parents[2] / "crates/uzu-engine/src/utils/version.rs"
    version = re.findall(r'TOOLCHAIN_VERSION: &str = "([^"]+)"', version_file.read_text())[0]
    with fetch(
        "https://sdk.trymirai.com/api/v1/fetch/models",
        {
            "device": {"os_name": "macOS", "cpu_name": None, "memory_total": None},
            "backends": [{"identifier": "uzu", "version": version}],
            "include_traces": False,
            "show_all": True,
        },
    ) as response:
        models = json.load(response)["models"]
    result = {}
    for model in models:
        reference = model["accessibility"]["reference"]
        name = reference["repository"]["identifier"]
        config_url = next(file["url"] for file in reference["files"] if file["name"] == "config.json")
        with fetch(config_url) as response:
            generation = json.load(response)["generation_config"]
        unsupported = {
            key: generation[key]
            for key in ("banned_tokens", "presence_penalty", "frequency_penalty")
            if generation.get(key) is not None
        }
        if unsupported:
            raise ValueError(f"{name}: unsupported sampling controls: {unsupported}")
        sampling = {
            key: generation.get(source)
            for key, source in (
                ("temp", "temperature"),
                ("top_k", "top_k"),
                ("top_p", "top_p"),
                ("min_p", "min_p"),
                ("repetition_penalty", "repetition_penalty"),
                ("suffix_repetition_length", "suffix_repetition_length"),
            )
        }
        specdec = any(file["name"].startswith("speculator/") for file in reference["files"])
        result[name] = {
            "model": name,
            "specdec": specdec,
            "sampling": sampling,
            "reasoning_efforts": reasoning_efforts(model["encodings"]),
        }
    write_catalog(directory, result)


if __name__ == "__main__":
    update()
