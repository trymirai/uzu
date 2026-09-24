#!/usr/bin/env python3
"""Convert a 0.16.0 model directory to the 0.17.0 config and weight layouts.

Requires NumPy. With no output directory, replaces the input after converting
it in a sibling directory; this needs free space for a full model copy.

    python scripts/convert_model_0_16_0_to_0_17_0.py MODEL_DIR [OUTPUT_DIR]
    python scripts/convert_model_0_16_0_to_0_17_0.py --check MODEL_DIR
"""

import argparse
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np


DTYPES = {"BF16": np.dtype("<u2"), "F32": np.dtype("<f4")}


def add_conv_configs(value):
    if isinstance(value, list):
        for item in value:
            add_conv_configs(item)
    elif isinstance(value, dict):
        if "layer_configs" in value:
            layers = value["layer_configs"]
            if not isinstance(layers, list):
                raise ValueError("Expected layer_configs to be an array")
            for layer in layers:
                if (
                    not isinstance(layer, dict)
                    or "mixer_config" not in layer
                    or "mlp_config" not in layer
                ):
                    raise ValueError("Expected a transformer layer in layer_configs")
                layer.setdefault("mixer_conv_config", None)
                layer.setdefault("mlp_conv_config", None)
        for item in value.values():
            add_conv_configs(item)


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
            continue
        if layout not in ("output_input", "input_output"):
            raise ValueError(f"unexpected layout in {key}: {layout}")
        if layout == "input_output":
            continue

        prefix = key.removesuffix(".spec")
        scale_name = prefix + ".scales"
        if scale_name not in header:
            continue
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

    array = np.frombuffer(data, dtype=np.uint8 if bits == 8 else DTYPES[dtype]).reshape(
        rows, groups
    )
    transposed = np.zeros((groups, padded_rows), dtype=array.dtype)
    transposed[:, :rows] = array.T
    return transposed.tobytes()


def convert_file(source_path, target_path, check):
    header, data_start = read_header(source_path)
    plan = conversion_plan(header)
    print(f"{source_path}: {len(plan)} parameter planes to transpose")
    if check:
        return

    ordered = sorted(
        (name for name in header if name != "__metadata__"),
        key=lambda name: header[name]["data_offsets"][0],
    )
    new_header = {"__metadata__": header.get("__metadata__", {})}
    offset = 0
    for name in ordered:
        tensor = header[name].copy()
        old_start, old_end = tensor["data_offsets"]
        if name in plan:
            target_shape, _, _, _ = plan[name]
            tensor["shape"] = target_shape
            item_size = (
                1
                if tensor["dtype"] == "U8"
                else np.dtype(DTYPES[tensor["dtype"]]).itemsize
            )
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
                _, bits, rows, groups = plan[name]
                data = source.read(remaining)
                converted = transpose_plane(
                    data, header[name]["dtype"], bits, rows, groups
                )
                assert (
                    len(converted)
                    == new_header[name]["data_offsets"][1]
                    - new_header[name]["data_offsets"][0]
                )
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate config and list planned weight changes without writing",
    )
    parser.add_argument("source", type=Path, help="0.16.0 model directory")
    parser.add_argument(
        "target", type=Path, nargs="?", help="new 0.17.0 model directory"
    )
    args = parser.parse_args()
    source = args.source.resolve()
    if args.source.is_symlink() or not source.is_dir():
        parser.error("source must be a model directory, not a symlink")
    destination = args.target.resolve() if args.target is not None else source
    if args.target is not None:
        if (
            destination == source
            or source in destination.parents
            or destination in source.parents
        ):
            parser.error("source and target must be separate directories")
        if (
            args.target.is_symlink()
            or destination.exists()
            or not destination.parent.is_dir()
        ):
            parser.error("target must be a new directory under an existing parent")

    config_path = source / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Expected one config object")
        add_conv_configs(config)
    except (OSError, ValueError) as error:
        parser.error(f"{config_path}: {error}")

    entries = sorted(source.rglob("*"))
    if any(path.is_symlink() for path in entries):
        parser.error("symlinks are not supported")
    files = [path for path in entries if path.is_file()]
    if not any(path.suffix == ".safetensors" for path in files):
        parser.error("no safetensors files found")
    if args.check:
        for path in files:
            if path.suffix == ".safetensors":
                convert_file(path, None, True)
        print(f"{config_path}: transformer convolution fields validated")
        return

    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.0.17.0.", dir=destination.parent)
    )
    staged = staging_root / "model"
    backup = staging_root / "backup"
    try:
        for path in files:
            relative = path.relative_to(source)
            output = staged / relative
            if path.suffix == ".safetensors":
                convert_file(path, output, False)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                if relative == Path("config.json"):
                    output.write_text(
                        json.dumps(config, indent=2) + "\n", encoding="utf-8"
                    )
                    shutil.copymode(path, output)
                else:
                    shutil.copy2(path, output)
        if args.target is not None:
            staged.rename(destination)
        else:
            source.rename(backup)
            try:
                staged.rename(source)
            except OSError:
                backup.rename(source)
                raise
            shutil.rmtree(backup)
    finally:
        if backup.exists():
            print(f"Original model retained at {backup}", file=sys.stderr)
        else:
            shutil.rmtree(staging_root)


if __name__ == "__main__":
    main()
