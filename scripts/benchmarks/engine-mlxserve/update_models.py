#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_models import catalog, fetch, upstream, write_catalog


def update():
    source = upstream("mlxserve", "src/host/cli/cli.zig").split("pub const aliases =", 1)[1].split("\n};", 1)[0]
    sidecars = upstream("mlxserve", "src/sdk/spec/mtp/source.zig").split("sidecar_rel_paths =", 1)[1].split("};", 1)[0]
    sidecars = set(re.findall(r'"([^"]+)"', sidecars))
    models = {}
    for line in source.splitlines():
        match = re.search(r'\.repo = "([^"]+)"', line)
        if not match or ".gguf_file =" in line:
            continue
        with fetch(f"https://huggingface.co/api/models/{match[1]}") as response:
            info = json.load(response)
        if info.get("pipeline_tag") in (None, "text-generation", "image-text-to-text"):
            models[match[1]] = any(file["rfilename"] in sidecars for file in info["siblings"])
    # mlx-serve decodes with a model's MTP head; the repository listing shows only a head kept in a sidecar file.
    write_catalog(
        Path(__file__).resolve().parent,
        catalog(models),
        speculation=lambda config: {"specdec": models[config["model"]] or "draft_model" in config},
        pin_revisions=True,
    )


if __name__ == "__main__":
    update()
