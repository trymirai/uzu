#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_models import hub_catalog, write_catalog

SAMPLING = ("temp", "top_k", "top_p", "min_p", "repetition_penalty", "suffix_repetition_length")


def update():
    write_catalog(
        Path(__file__).resolve().parent,
        hub_catalog("mlx-community", "LiquidAI"),
        speculation=lambda config: {"specdec": False},
        pin_revisions=True,
    )


if __name__ == "__main__":
    update()
