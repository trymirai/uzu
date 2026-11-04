#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["packaging>=26.3"]
# ///

import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_engines import latest_release, parse_args, update_lockfile


def update(*, dry_run=False):
    directory = Path(__file__).resolve().parent
    source = tomllib.loads((directory / "pyproject.toml").read_text())["tool"]["uv"]["sources"]["omlx"]
    repository = source["git"].removeprefix("https://github.com/").removesuffix(".git")
    tag = latest_release(repository)
    print(f"omlx: {source['tag']} -> {tag}", flush=True)
    if not dry_run:
        subprocess.run(
            ["uv", "add", "--project", str(directory), "--frozen", "--tag", tag, f"omlx @ git+{source['git']}"],
            check=True,
        )
    update_lockfile(directory, dry_run=dry_run)


if __name__ == "__main__":
    update(dry_run=parse_args().dry_run)
