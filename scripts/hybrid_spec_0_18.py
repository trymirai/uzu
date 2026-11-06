#!/usr/bin/env python3
"""Rewrite a model exported for format 0.17 to format 0.18: flat hybrid incoherence fields become a nested incoherence_spec.

Only safetensors header metadata changes; tensor bytes are copied unchanged, other files are hard-linked.
usage: hybrid_spec_0_18.py SOURCE_DIR OUTPUT_DIR
"""

import json
import os
import shutil
import struct
import sys
from pathlib import Path


def nested(spec: dict) -> dict:
    assert spec.get("incoherence_kind", "hadamard") == "hadamard", f"format 0.18 reads no {spec['incoherence_kind']} rotation"
    block_size = spec["incoherence_block_size"]
    rotation = {"type": "HadamardSpec", "block_size": block_size, "processing_mode": spec["incoherence_processing_mode"]}
    return {
        "type": "HybridSpec",
        "quantization_spec": convert(spec["quantization_spec"]),
        "adapter_spec": spec["adapter_spec"],
        "incoherence_spec": None if block_size is None else rotation,
    }


def convert(spec: dict) -> dict:
    return nested(spec) if spec["type"] == "HybridSpec" else spec


def convert_file(source: Path, output: Path) -> int:
    with source.open("rb") as reader, output.open("wb") as writer:
        header = json.loads(reader.read(struct.unpack("<Q", reader.read(8))[0]))
        metadata = header.get("__metadata__", {})
        specs = [key for key, value in metadata.items() if '"HybridSpec"' in value]
        for key in specs:
            metadata[key] = json.dumps(convert(json.loads(metadata[key])))
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)  # tensor offsets are relative to the end of the header; the format pads it
        writer.write(struct.pack("<Q", len(encoded)) + encoded)
        shutil.copyfileobj(reader, writer, 1 << 24)
    return len(specs)


def main() -> None:
    assert len(sys.argv) == 3, __doc__
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    assert not output.exists(), f"{output} already exists"
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".safetensors":
            print(f"{path.relative_to(source)}: {convert_file(path, target)} hybrid specs")
        else:
            os.link(path, target)


if __name__ == "__main__":
    main()
