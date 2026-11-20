#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["packaging>=26.3"]
# ///

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_engines import parse_args, update_sources


def update(*, dry_run=False):
    update_sources("llama_cpp", "cli11", dry_run=dry_run)


if __name__ == "__main__":
    update(dry_run=parse_args().dry_run)
