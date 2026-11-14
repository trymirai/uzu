#!/usr/bin/env python3

"""Refresh engine catalogs without downloading weights or importing engines."""

import json
import re
import sys
import tempfile
from functools import cache
from importlib import import_module
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def fetch(url, payload=None):
    if payload is not None:
        url = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return urlopen(url, timeout=60)


def upstream(engine, path):
    source = json.loads((ROOT / "sources.json").read_text())[engine]
    url = f"https://raw.githubusercontent.com/{source['owner']}/{source['repo']}/{source['rev']}/{path}"
    with fetch(url) as response:
        return response.read().decode()


def hub_models(**query):
    url = "https://huggingface.co/api/models?" + urlencode({"limit": 1000, **query})
    while url:
        with fetch(url) as response:
            yield from json.load(response)
            link = re.search(r'<([^>]+)>; rel="next"', response.headers.get("Link", ""))
            url = link[1] if link else None


def catalog(models):
    return {model: {"model": model} for model in models}


def hub_catalog(*authors):
    return catalog(
        model["id"]
        for author in authors
        for model in hub_models(author=author)
        if model.get("pipeline_tag") in (None, "text-generation", "image-text-to-text")
    )


@cache
def hub_revision(repository):
    with fetch(f"https://huggingface.co/api/models/{repository}/revision/main") as response:
        return json.load(response)["sha"]


REVISION_FIELDS = {"model": "revision", "draft_model": "draft_revision"}


def pin_revisions_of(config):
    pinned = {}
    for field, value in config.items():
        pinned[field] = value
        if field in REVISION_FIELDS:
            pinned[REVISION_FIELDS[field]] = hub_revision(value.partition(":")[0])
    return pinned


def write_catalog(directory, models, *, speculation=None, pin_revisions=False):
    """`speculation` returns how the engine speculates with a mapped configuration: `specdec` and its own fields."""
    if not models:
        raise ValueError(f"Empty upstream catalog for {directory.name}")
    mapping = directory / "model_mapping.json"
    if directory.name != "engine-uzu":
        targets = json.loads(mapping.read_text())
        targets = {name: {"model": target} if isinstance(target, str) else target for name, target in targets.items()}
        missing = {
            f"{name}.{field}": config[field]
            for name, config in targets.items()
            for field in ("model", "draft_model")
            if config.get(field) is not None and config[field] not in models
        }
        if missing:
            raise ValueError(f"{mapping}: mapped models disappeared from upstream:\n{json.dumps(missing, indent=2)}")
        uzu_models = json.loads((directory.parent / "engine-uzu/models.json").read_text())
        resolved = {}
        for name, config in targets.items():
            config = models[config.pop("model")] | config
            uzu = {field: uzu_models[name][field] for field in ("sampling", "reasoning_efforts")}
            resolved[name] = config | speculation(config) | uzu
        models = resolved
    if pin_revisions:
        models = {name: pin_revisions_of(config) for name, config in models.items()}
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as file:
        file.write(json.dumps(dict(sorted(models.items())), indent=2) + "\n")
    Path(file.name).replace(directory / "models.json")
    print(f"{directory.name}: {len(models)} models", file=sys.stderr)


if __name__ == "__main__":
    for directory in sorted(ROOT.glob("engine-*"), key=lambda path: (path.name != "engine-uzu", path.name)):
        import_module(f"{directory.name}.update_models").update()
