#!/usr/bin/env python3

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_models import catalog, hub_catalog, upstream, write_catalog


def update():
    source = upstream("tensorfold", "app/electron/agent/engine/catalog.ts").split("SUGGESTED_MODELS:", 1)[1]
    models = hub_catalog("mlx-community", "LiquidAI", "incoai") | catalog(re.findall(r"repo: '([^']+)'", source))
    write_catalog(
        Path(__file__).resolve().parent,
        models,
        speculation=lambda config: {"specdec": "draft_model" in config},
        pin_revisions=True,
    )


if __name__ == "__main__":
    update()
