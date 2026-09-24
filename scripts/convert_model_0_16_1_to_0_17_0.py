#!/usr/bin/env python3
"""Convert a 0.16.1 model directory to the 0.17.0 quantization-parameter layout.

Requires NumPy. The source is never modified. Each safetensors file is streamed,
so model weights do not need to fit in memory. Configs are copied unchanged;
0.16.1 already supplies the transformer conv and reasoning fields.

    bash scripts/convert_model_0_16_1_to_0_17_0.sh MODEL_DIR [OUTPUT_DIR]
    python scripts/convert_model_0_16_1_to_0_17_0.py --check MODEL_DIR
"""

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np


DTYPES = {"BF16": np.dtype("<u2"), "F32": np.dtype("<f4")}


def read_header(path):
    with path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
    return header, 8 + header_size


def conversion_plan(header):
    metadata = header.get("__metadata__", {})
    plan = {}
    for key, raw_spec in metadata.items():
        if not key.endswith(".spec"):
            continue
        spec = json.loads(raw_spec)
        layout = spec.get("layout")
        if layout is None:
            continue  # Composite specs have their own child matrix specs.
        if layout not in ("output_input", "input_output"):
            raise ValueError(f"unexpected layout in {key}: {layout}")
        if layout == "input_output":
            continue  # Embedding and tied readout retain [N, G].

        prefix = key.removesuffix(".spec")
        scale_name = prefix + ".scales"
        if scale_name not in header:
            continue  # Full-precision matrices have no quantization parameters.
        rows, groups = header[scale_name]["shape"]
        if rows <= 0 or groups <= 0:
            raise ValueError(f"invalid scale shape for {scale_name}")
        padded_rows = (rows + 3) // 4 * 4
        for suffix in ("scales", "biases", "zero_points"):
            name = prefix + "." + suffix
            if name not in header:
                continue
            tensor = header[name]
            if suffix == "zero_points":
                bits = spec["bits"]
                if bits not in (4, 8) or tensor["dtype"] != "U8":
                    raise ValueError(f"unsupported zero points in {name}")
                packed_groups = (groups * bits + 7) // 8
                if tensor["shape"] != [rows, packed_groups]:
                    raise ValueError(f"unexpected shape for {name}: {tensor['shape']}")
                target_shape = [groups, padded_rows * bits // 8]
            else:
                if tensor["dtype"] not in DTYPES or tensor["shape"] != [rows, groups]:
                    raise ValueError(f"unexpected shape or dtype for {name}: {tensor}")
                bits = None
                target_shape = [groups, padded_rows]
            plan[name] = (target_shape, bits, rows, groups)
    return plan


def transpose_plane(data, dtype, bits, rows, groups):
    padded_rows = (rows + 3) // 4 * 4
    if bits == 4:
        packed = np.frombuffer(data, dtype=np.uint8).reshape(rows, (groups + 1) // 2)
        unpacked = np.empty((rows, packed.shape[1] * 2), dtype=np.uint8)
        unpacked[:, 0::2] = packed & 15
        unpacked[:, 1::2] = packed >> 4
        transposed = np.zeros((groups, padded_rows), dtype=np.uint8)
        transposed[:, :rows] = unpacked[:, :groups].T
        return (transposed[:, 0::2] | (transposed[:, 1::2] << 4)).tobytes()

    array = np.frombuffer(data, dtype=np.uint8 if bits == 8 else DTYPES[dtype]).reshape(rows, groups)
    transposed = np.zeros((groups, padded_rows), dtype=array.dtype)
    transposed[:, :rows] = array.T
    return transposed.tobytes()


def convert_file(source_path, target_path, check):
    header, data_start = read_header(source_path)
    plan = conversion_plan(header)
    print(f"{source_path}: {len(plan)} parameter planes to transpose")
    if check:
        return

    ordered = sorted((name for name in header if name != "__metadata__"), key=lambda name: header[name]["data_offsets"][0])
    new_header = {"__metadata__": header.get("__metadata__", {})}
    offset = 0
    for name in ordered:
        tensor = header[name].copy()
        old_start, old_end = tensor["data_offsets"]
        if name in plan:
            target_shape, _, _, _ = plan[name]
            tensor["shape"] = target_shape
            item_size = 1 if tensor["dtype"] == "U8" else np.dtype(DTYPES[tensor["dtype"]]).itemsize
            size = target_shape[0] * target_shape[1] * item_size
        else:
            size = old_end - old_start
        tensor["data_offsets"] = [offset, offset + size]
        new_header[name] = tensor
        offset += size

    encoded = json.dumps(new_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as source, target_path.open("xb") as target:
        target.write(struct.pack("<Q", len(encoded)))
        target.write(encoded)
        for name in ordered:
            old_start, old_end = header[name]["data_offsets"]
            source.seek(data_start + old_start)
            remaining = old_end - old_start
            if name in plan:
                target_shape, bits, rows, groups = plan[name]
                data = source.read(remaining)
                converted = transpose_plane(data, header[name]["dtype"], bits, rows, groups)
                assert len(converted) == new_header[name]["data_offsets"][1] - new_header[name]["data_offsets"][0]
                target.write(converted)
            else:
                while remaining:
                    chunk = source.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f"truncated tensor {name} in {source_path}")
                    target.write(chunk)
                    remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate headers and list planned changes without writing")
    parser.add_argument("source", type=Path, help="0.16.1 model directory")
    parser.add_argument("target", type=Path, nargs="?", help="new 0.17.0 model directory")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        parser.error("source must be a model directory")
    if not args.check and args.target is None:
        parser.error("target is required without --check")
    if args.target is not None:
        target = args.target.resolve()
        if target == source or source in target.parents or target in source.parents:
            parser.error("source and target must be separate directories")
        if target.exists():
            parser.error("target already exists")
    entries = sorted(source.rglob("*"))
    if any(path.is_symlink() for path in entries):
        parser.error("symlinks are not supported")
    files = [path for path in entries if path.is_file()]
    if not any(path.suffix == ".safetensors" for path in files):
        parser.error("no safetensors files found")
    for path in files:
        relative = path.relative_to(source)
        if path.suffix == ".safetensors":
            convert_file(path, None if args.check else target / relative, args.check)
        elif not args.check:
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / relative)


if __name__ == "__main__":
    main()
