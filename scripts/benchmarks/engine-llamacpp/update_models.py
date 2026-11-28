#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_models import gguf_catalog, write_catalog

# llama.cpp speculates only to an explicit depth; production ran every MTP and DFlash model at 2.
SPECULATIVE_DEPTH = 2

SAMPLING = ("temp", "top_k", "top_p", "min_p", "repetition_penalty", "suffix_repetition_length")


def speculation(config):
    # Unsloth publishes GGUFs that keep their MTP heads in -MTP- repositories.
    if "draft_model" in config or "-MTP-" in config["model"]:
        return {"speculative_depth": SPECULATIVE_DEPTH, "specdec": True}
    return {"specdec": False}


def update():
    write_catalog(Path(__file__).resolve().parent, gguf_catalog("unsloth", "LiquidAI"), speculation=speculation)


if __name__ == "__main__":
    update()
