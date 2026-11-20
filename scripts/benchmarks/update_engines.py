#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["packaging>=26.3"]
# ///

"""Update external engine pins and lockfiles without building or installing engines."""

import argparse
import json
import subprocess
from importlib import import_module
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parent


def fetch_json(url):
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def latest_release(repository):
    page = 1
    while releases := fetch_json(f"https://api.github.com/repos/{repository}/releases?per_page=100&page={page}"):
        for release in releases:
            if release["draft"] or release["prerelease"]:
                continue
            tag = release["tag_name"]
            try:
                version = Version(tag.rsplit("@", 1)[-1])
            except InvalidVersion:
                continue
            if not version.is_prerelease:
                return tag
        page += 1
    raise ValueError(f"{repository}: no stable release found")


def source_hash(source):
    repository = f"{source['owner']}/{source['repo']}"
    if source.get("fetchSubmodules"):
        query = urlencode({"ref": f"refs/tags/{source['rev']}", "submodules": 1, "shallow": 1})
        url = f"git+https://github.com/{repository}.git?{query}"
    else:
        url = f"github:{repository}?{urlencode({'ref': source['rev']})}"
    result = subprocess.check_output(["nix", "flake", "prefetch", "--json", url], text=True)
    return json.loads(result)["hash"]


def update_sources(*names, branch=None, dry_run=False):
    sources_file = ROOT / "sources.json"
    sources = json.loads(sources_file.read_text())
    changed = []
    for name in names:
        source = sources[name]
        repository = f"{source['owner']}/{source['repo']}"
        revision = (
            fetch_json(f"https://api.github.com/repos/{repository}/commits/{branch}")["sha"]
            if branch is not None
            else latest_release(repository)
        )
        print(f"{name}: {source['rev']} -> {revision}", flush=True)
        if revision != source["rev"]:
            source["rev"] = revision
            changed.append(name)

    if dry_run:
        return

    for name in changed:
        sources[name]["hash"] = source_hash(sources[name])
    if changed:
        sources_file.write_text(json.dumps(sources, indent=2) + "\n")


def update_lockfile(directory, *, dry_run=False):
    if not dry_run:
        subprocess.run(["uv", "lock", "--project", str(directory), "--upgrade"], check=True)


def update_pypi(directory, *packages, dry_run=False):
    pins = []
    for package in packages:
        version = fetch_json(f"https://pypi.org/pypi/{package}/json")["info"]["version"]
        pins.append(f"{package}=={version}")
    print(f"{directory.name.removeprefix('engine-')}: {', '.join(pins)}", flush=True)
    if not dry_run:
        subprocess.run(["uv", "add", "--project", str(directory), "--frozen", *pins], check=True)
    update_lockfile(directory, dry_run=dry_run)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show proposed engine pins without changing files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import_module("common-cpp.update_dependencies").update(dry_run=args.dry_run)
    for script in sorted(ROOT.glob("engine-*/update_engine.py")):
        import_module(f"{script.parent.name}.update_engine").update(dry_run=args.dry_run)
    if not args.dry_run:
        print("Pins updated. Review patches and build the engines before benchmarking.")
