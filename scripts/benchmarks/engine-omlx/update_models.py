#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_models import hub_catalog, write_catalog


def update():
    write_catalog(
        Path(__file__).resolve().parent,
        hub_catalog("mlx-community", "LiquidAI", "z-lab", "incoai"),
        speculation=lambda config: {"specdec": "draft_model" in config},
        pin_revisions=True,
    )


if __name__ == "__main__":
    update()
