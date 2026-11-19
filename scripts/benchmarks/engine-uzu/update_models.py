#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common-py/src"))
from reasoning import map_reasoning_efforts

from update_models import common_sampling, fetch, write_catalog

HANASHI = Path(__file__).resolve().parents[3] / "crates/legacy/hanashi/configs"


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
    # As RenderingConfig::default_reasoning_effort: the one effort that renders the same as the default.
    if default == "default" and any(mapping.get("default") is not None for mapping in mappings):
        same = [
            effort
            for effort in supported - {"default"}
            if all(mapping[effort] == mapping["default"] for mapping in mappings)
        ]
        if len(same) == 1:
            (default,) = same
    return map_reasoning_efforts(supported, default)


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
        if not any(specialization["type"] == "chat" for specialization in model["specializations"]):
            continue
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
            "model_size": sum(file["size"] for file in reference["files"]),
            "specdec": specdec,
            "sampling": common_sampling(name, sampling),
            "reasoning_efforts": reasoning_efforts(model["encodings"]),
        }
    write_catalog(directory, result)


if __name__ == "__main__":
    update()
