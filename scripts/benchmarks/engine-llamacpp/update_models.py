#!/usr/bin/env python3

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_models import catalog, hub_models, write_catalog

# llama.cpp speculates only to an explicit depth; production ran every MTP and DFlash model at 2.
SPECULATIVE_DEPTH = 2


def speculation(config):
    # Unsloth publishes GGUFs that keep their MTP heads in -MTP- repositories.
    if "draft_model" in config or "-MTP-" in config["model"]:
        return {"speculative_depth": SPECULATIVE_DEPTH, "specdec": True}
    return {"specdec": False}


def update():
    models = set()
    repos = (
        repo
        for author in ("unsloth", "LiquidAI")
        for repo in hub_models(author=author, filter="gguf", **{"expand[]": "siblings"})
    )
    for repo in repos:
        for file in repo["siblings"]:
            name = Path(file["rfilename"]).name
            if name.lower().startswith("dflash") and name.endswith(".gguf"):
                models.add(f"{repo['id']}:{file['rfilename']}")
            if not name.endswith(".gguf") or name.lower().startswith(("mmproj", "mtp", "dflash")):
                continue
            quant = re.search(r"[.-]((?:UD-)?(?:I?Q\d_[\w]+|BF16|F16|FP16|MXFP4))(?:-\d+-of-\d+)?\.gguf$", name)
            if quant:
                models.add(f"{repo['id']}:{quant[1]}")
    write_catalog(Path(__file__).resolve().parent, catalog(models), speculation=speculation)


if __name__ == "__main__":
    update()
