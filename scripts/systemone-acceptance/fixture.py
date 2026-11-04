"""Pinned, public Liquid fixture. No inference and no implicit download."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate(model):
    manifest = json.loads((HERE / "liquid.json").read_text())
    if not model.is_dir():
        raise ValueError(f"required model directory does not exist: {model}")
    for name, expected in manifest["files"].items():
        path = model / name
        if not path.is_file() or digest(path) != expected["sha256"]:
            raise ValueError(f"missing or mismatched pinned fixture file: {path}")
    return manifest


def prepare(model):
    manifest = json.loads((HERE / "liquid.json").read_text())
    model.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["files"].items():
        path = model / name
        if path.exists():
            if not path.is_file() or digest(path) != expected["sha256"]:
                raise ValueError(f"refusing to overwrite mismatched fixture file: {path}")
            continue
        temporary = path.with_suffix(path.suffix + ".download")
        if temporary.exists():
            raise ValueError(f"unfinished download exists; inspect it first: {temporary}")
        if "content" in expected:
            temporary.write_text(expected["content"])
        else:
            url = expected.get("url") or (
                f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{name}"
            )
            with urllib.request.urlopen(url, timeout=120) as source, temporary.open("xb") as destination:
                while block := source.read(1024 * 1024):
                    destination.write(block)
        if digest(temporary) != expected["sha256"]:
            raise ValueError(f"download checksum mismatch: {temporary}")
        temporary.replace(path)
    validate(model)
    print(json.dumps({"fixture": "PASS", "model": str(model), "revision": manifest["revision"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model.resolve())
