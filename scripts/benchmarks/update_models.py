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


def gguf_catalog(*authors):
    models = set()
    repos = (
        repo for author in authors for repo in hub_models(author=author, filter="gguf", **{"expand[]": "siblings"})
    )
    for repo in repos:
        for file in repo["siblings"]:
            name = Path(file["rfilename"]).name
            if name.lower().startswith("dflash") and name.endswith(".gguf"):
                models.add(f"{repo['id']}:{file['rfilename']}")
            if not name.endswith(".gguf") or name.lower().startswith(("mmproj", "mtp", "dflash")):
                continue
            quant = re.search(r"[.-]((?:UD-)?(?:I?Q\d_[\w]+|BF16|F16|FP16|MXFP4))(?:-\d+-of-\d+)?\.gguf$", name)
            if quant:
                models.add(f"{repo['id']}:{quant[1]}")
    return catalog(models)


@cache
def hub_revision(repository):
    with fetch(f"https://huggingface.co/api/models/{repository}/revision/main") as response:
        return json.load(response)["sha"]


REVISION_FIELDS = {"model": "revision", "draft_model": "draft_revision"}


@cache
def hub_files(repository, revision):
    with fetch(f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true") as response:
        return {file["rfilename"]: file["size"] for file in json.load(response)["siblings"]}


def hub_size(model, revision):
    """Bytes of a repository, of `repo:file`, or of the GGUF quantization `repo:QUANT`."""
    repository, _, variant = model.partition(":")
    files = hub_files(repository, revision or "main")
    if not variant:
        return sum(files.values())
    if variant in files:
        return files[variant]
    # The Hub manifest names the file llama.cpp downloads for a quantization, possibly the first of several parts.
    manifest = f"https://huggingface.co/v2/{repository}/manifests/{variant}"
    with fetch(Request(manifest, headers={"User-Agent": "llama-cpp"})) as response:
        first = json.load(response)["ggufFile"]["rfilename"]
    split = re.fullmatch(r"(.*)-\d+-of-(\d+\.gguf)", first)
    parts = re.escape(split[1]) + r"-\d+-of-" + re.escape(split[2]) if split else re.escape(first)
    return sum(size for name, size in files.items() if re.fullmatch(parts, name))


def sized(config, size):
    """The configuration with `model_size` after the model and its revision."""
    after = "revision" if "revision" in config else "model"
    result = {}
    for field, value in config.items():
        if field != "model_size":
            result[field] = value
        if field == after:
            result["model_size"] = size
    return result


def model_size(config):
    """Bytes of the model and its draft at their pinned revisions."""
    return sum(
        hub_size(config[field], config.get(revision)) for field, revision in REVISION_FIELDS.items() if field in config
    )


def pin_revisions_of(config):
    pinned = {}
    for field, value in config.items():
        pinned[field] = value
        if field in REVISION_FIELDS:
            pinned[REVISION_FIELDS[field]] = hub_revision(value.partition(":")[0])
    return pinned


def common_sampling(name, sampling):
    """The sampling without parameters that an engine with the model doesn't apply, so all engines sample alike."""
    engines = {
        mapping.parent.name: import_module(f"{mapping.parent.name}.update_models").SAMPLING
        for mapping in sorted(ROOT.glob("engine-*/model_mapping.json"))
        if name in json.loads(mapping.read_text())
    }
    common = {}
    for parameter, value in sampling.items():
        without = [engine for engine, applied in engines.items() if parameter not in applied]
        common[parameter] = None if without else value
        if value is not None and without:
            print(f"{name}: without {parameter} {value}, not applied by {', '.join(without)}", file=sys.stderr)
    return common


def write_catalog(directory, models, *, speculation=None, pin_revisions=False):
    """`speculation` maps a configuration to the engine's speculation fields, `specdec` included."""
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
    if directory.name != "engine-uzu":
        models = {name: sized(config, model_size(config)) for name, config in models.items()}
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as file:
        file.write(json.dumps(dict(sorted(models.items())), indent=2) + "\n")
    Path(file.name).replace(directory / "models.json")
    print(f"{directory.name}: {len(models)} models", file=sys.stderr)


if __name__ == "__main__":
    for directory in sorted(ROOT.glob("engine-*"), key=lambda path: (path.name != "engine-uzu", path.name)):
        import_module(f"{directory.name}.update_models").update()
