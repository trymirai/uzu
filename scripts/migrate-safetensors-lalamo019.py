#!/usr/bin/env python3
"""Rewrite safetensors header keys for the lalamo 0.19 transformer-layer schema.

Upstream 7568c26b moved body weights under `mixer.body` / `mlp.body` and nested
the norms (`pre_mixer_norm` -> `mixer.pre_norm`, etc.). Old local bundles store
both tensor names and `__metadata__` spec keys with the flat paths.

Applies to keys under `decoder.transformer.layers.<N>/`:
  mixer.<x>            -> mixer.body.<x>        (body projections/conv/qk-norms)
  mlp.<x>              -> mlp.body.<x>          (up/gate/down/router/experts)
  pre_mixer_norm.<x>   -> mixer.pre_norm.<x>
  post_mixer_norm.<x>  -> mixer.post_norm.<x>
  pre_mlp_norm.<x>     -> mlp.pre_norm.<x>
  post_mlp_norm.<x>    -> mlp.post_norm.<x>

NOTE: if a bundle carries a layer-level conv (`mixer_conv.*`/`mlp_conv.*`) this
script must be extended — old body-conv (`mixer.conv.*`) and a migrated layer
conv would collide under the same prefix. The three bundles this was written
for (LFM3-VL-2B, LFM3-1.7B, MiniCPM-V-4.7 MoE) have no layer-level conv.

Data blobs are copied verbatim; only the JSON header changes, so tensor offsets
stay valid. Run scripts/migrate-config-lalamo019.py first.

Usage: python3 scripts/migrate-safetensors-lalamo019.py <bundle-dir> [more...]
"""

import json
import re

import os
import struct
import sys
from pathlib import Path

LAYER_RE = re.compile(r"^(decoder\.transformer\.layers\.\d+\.)(.*)$")
RENAMES = [
    ("pre_mixer_norm.", "mixer.pre_norm."),
    ("post_mixer_norm.", "mixer.post_norm."),
    ("pre_mlp_norm.", "mlp.pre_norm."),
    ("post_mlp_norm.", "mlp.post_norm."),
]


def new_key(key: str) -> str:
    match = LAYER_RE.match(key)
    if not match:
        return key
    prefix, rest = match.groups()
    for old, new in RENAMES:
        if rest.startswith(old):
            return prefix + new + rest[len(old):]
        if rest.startswith(new):
            return key  # already migrated
    if rest.startswith("mixer_conv."):
        return prefix + "mixer.conv." + rest[len("mixer_conv."):]
    if rest.startswith("mlp_conv."):
        return prefix + "mlp.conv." + rest[len("mlp_conv."):]
    if rest.startswith("mixer.body.") or rest.startswith("mlp.body."):
        return key  # already migrated
    if rest.startswith("mixer."):
        return prefix + "mixer.body." + rest[len("mixer."):]
    if rest.startswith("mlp."):
        return prefix + "mlp.body." + rest[len("mlp."):]
    return key


def transform_header(header: dict) -> dict:
    """Rename tensor keys and the `__metadata__` spec keys nested inside."""
    new_header = {key if key == "__metadata__" else new_key(key): value for key, value in header.items()}
    metadata = new_header.get("__metadata__")
    if metadata:
        new_header["__metadata__"] = {new_key(k): v for k, v in metadata.items()}
    return new_header


def rename_collides(header: dict) -> bool:
    renamed = {new_key(k) for k in header if k != "__metadata__"}
    return len(renamed) != len(header) - 1


def migrate_file(path: Path) -> None:
    buf = path.read_bytes()
    header_len = struct.unpack("<Q", buf[:8])[0]
    header = json.loads(buf[8:8 + header_len])
    new_header = transform_header(header)
    if new_header == header:
        print(f"{path}: already migrated")
        return
    if rename_collides(new_header):
        sys.exit(f"{path}: key collision after rename, aborting")
    blob = json.dumps(new_header, separators=(",", ":")).encode()
    # The header grows, so the data start shifts; rewrite via RAM to avoid a
    # second full-size copy on a nearly-full disk. Original is removed before
    # writing — re-derive from HF if this crashes mid-write.
    target = path.with_suffix(".safetensors.tmp")
    view = memoryview(buf)
    os.remove(path)
    try:
        with target.open("wb") as out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            out.write(view[8 + header_len:])
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    if target.stat().st_size != 8 + len(blob) + len(buf) - 8 - header_len:
        sys.exit(f"{path}: size mismatch after rewrite, aborting")
    target.replace(path)
    print(f"{path}: {len(header) - 1} tensors rewritten")


def migrate_bundle(bundle: Path) -> None:
    for path in sorted(bundle.glob("*.safetensors")):
        migrate_file(path)
    index = bundle / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        if "weight_map" in data:
            new_map = {new_key(k): v for k, v in data["weight_map"].items()}
            if len(new_map) != len(data["weight_map"]) or len(set(new_map)) != len(new_map):
                sys.exit(f"{index}: key collision after rename, aborting")
            if new_map != data["weight_map"]:
                data["weight_map"] = new_map
                index.write_text(json.dumps(data))
                print(f"{index}: weight_map renamed ({len(new_map)} keys)")
            else:
                print(f"{index}: already migrated")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        migrate_bundle(Path(arg).expanduser())


if __name__ == "__main__":
    main()
