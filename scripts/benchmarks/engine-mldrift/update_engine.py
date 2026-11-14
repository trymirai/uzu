#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["packaging>=26.3"]
# ///

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT.parent))
from update_engines import parse_args, update_sources


def update_requirements():
    source = json.loads((PROJECT.parent / "sources.json").read_text())["mldrift"]
    url = f"https://raw.githubusercontent.com/{source['owner']}/{source['repo']}/{source['rev']}/requirements.txt"
    with urlopen(url, timeout=30) as response, TemporaryDirectory() as temporary:
        Path(temporary, "requirements.txt").write_bytes(response.read())
        subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                "requirements.txt",
                "--python-version",
                "3.11",
                "--python-platform",
                "aarch64-apple-darwin",
                "--index-strategy",
                "unsafe-best-match",
                "--generate-hashes",
                "--emit-index-url",
                "--emit-build-options",
                "--custom-compile-command",
                "./update_engine.py",
                "--output-file",
                PROJECT / "requirements.lock",
            ],
            cwd=temporary,
            check=True,
        )


def update(*, dry_run=False):
    update_sources("mldrift", branch="main", dry_run=dry_run)
    if not dry_run:
        update_requirements()


if __name__ == "__main__":
    update(dry_run=parse_args().dry_run)
