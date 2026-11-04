#!/usr/bin/env python3

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_models import catalog, upstream, write_catalog


def update():
    data = json.loads(upstream("magnitude", "inference/catalog/models.json"))
    models = catalog(
        f"{source['repository']}:{variant['format']}"
        for model in data["models"]
        if model["support"]["level"] == "supported"
        for source in model["sources"]
        for variant in source["variants"]
    )
    # The adapter passes no draft, so Magnitude speculates only with MTP heads kept in the GGUF,
    # which unsloth publishes in -MTP- repositories.
    write_catalog(
        Path(__file__).resolve().parent,
        models,
        speculation=lambda config: {"specdec": "-MTP-" in config["model"]},
        pin_revisions=True,
    )


if __name__ == "__main__":
    update()
