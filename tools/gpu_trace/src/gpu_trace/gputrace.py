"""Parse .gputrace capture bundles.

Reads MTLBuffer/MTLTexture files from .gputrace bundles, extracts resource
labels from device-resources binary files, and reads buffer data with
configurable struct layouts.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaptureMetadata:
    uuid: str
    captured_frames: int
    graphics_api: str


@dataclass(frozen=True)
class CaptureResource:
    filename: str
    size_bytes: int
    label: str
    kind: str  # "buffer" or "texture"


@dataclass(frozen=True)
class ShaderLibrary:
    path: str


@dataclass(frozen=True)
class CaptureInfo:
    metadata: CaptureMetadata
    resources: tuple[CaptureResource, ...]
    shader_functions: tuple[str, ...]
    shader_libraries: tuple[ShaderLibrary, ...]


@dataclass(frozen=True)
class BufferElement:
    index: int
    fields: tuple[tuple[str, object], ...]  # (field_name, value)


# ── metadata ──────────────────────────────────────────────────────────────


def parse_metadata(gputrace: Path) -> CaptureMetadata:
    meta_path = gputrace / "metadata"
    if not meta_path.exists():
        return CaptureMetadata(uuid="", captured_frames=0, graphics_api="")

    result = subprocess.run(
        ["plutil", "-convert", "json", "-o", "-", str(meta_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return CaptureMetadata(uuid="", captured_frames=0, graphics_api="")

    data = json.loads(result.stdout)
    api_code = data.get("DYCaptureSession.graphics_api")
    api = {1: "Metal"}.get(api_code, f"Unknown({api_code})" if api_code else "Unknown")

    return CaptureMetadata(
        uuid=data.get("(uuid)", ""),
        captured_frames=data.get("DYCaptureEngine.captured_frames_count", 0),
        graphics_api=api,
    )


# ── binary string extraction ─────────────────────────────────────────────


def _extract_strings(data: bytes, min_length: int = 3) -> list[tuple[int, str]]:
    """Extract printable ASCII strings with their byte offsets."""
    strings: list[tuple[int, str]] = []
    current: list[str] = []
    start: int | None = None

    for i, b in enumerate(data):
        if 32 <= b <= 126:
            if start is None:
                start = i
            current.append(chr(b))
        else:
            if len(current) >= min_length:
                assert start is not None
                strings.append((start, "".join(current)))
            current = []
            start = None

    if len(current) >= min_length and start is not None:
        strings.append((start, "".join(current)))

    return strings


# ── label extraction ──────────────────────────────────────────────────────


def extract_buffer_labels(gputrace: Path) -> dict[str, str]:
    """Extract MTLBuffer/MTLTexture filename → label mapping.

    Scans device-resources binary files for printable strings, matching
    resource filenames to their nearest following label string.
    """
    label_map: dict[str, str] = {}

    for fname in os.listdir(gputrace):
        if not fname.startswith("device-resources"):
            continue
        data = (gputrace / fname).read_bytes()
        strings = _extract_strings(data)

        resource_refs = [
            (off, s)
            for off, s in strings
            if re.match(r"MTL(Buffer|Texture)-\d+-\d+", s)
        ]

        label_strings = [
            (off, s)
            for off, s in strings
            if " " in s
            and len(s) >= 5
            and not s.startswith("MTL")
            and not s.startswith("/")
        ]

        for roff, rname in resource_refs:
            best_label: str | None = None
            best_dist = float("inf")
            for loff, lval in label_strings:
                dist = loff - roff
                if 0 < dist < best_dist:
                    best_dist = dist
                    best_label = lval
            if best_label:
                label_map[rname] = best_label

    return label_map


# ── shader info ───────────────────────────────────────────────────────────


def extract_shader_info(
    gputrace: Path,
) -> tuple[list[str], list[str]]:
    """Extract shader function names and library paths.

    Returns (functions, libraries).
    """
    functions: list[str] = []
    libraries: list[str] = []

    for fname in os.listdir(gputrace):
        if not fname.startswith("device-resources"):
            continue
        data = (gputrace / fname).read_bytes()
        strings = _extract_strings(data)

        for _, s in strings:
            if s.endswith(".metallib"):
                libraries.append(s)

        in_functions = False
        for _, s in strings:
            if s.endswith(".metallib"):
                in_functions = True
                continue
            if s == "pipeline-libraries":
                in_functions = False
                continue
            if (
                in_functions
                and re.match(r"^[a-z_][a-z0-9_]*$", s)
                and len(s) > 3
                and s not in ("function", "functions", "buffer", "buffers")
            ):
                if s not in functions:
                    functions.append(s)

    return functions, libraries


# ── buffer reading ────────────────────────────────────────────────────────

_LAYOUT_FORMATS: dict[str, tuple[str, int, int]] = {
    "float": ("f", 1, 1),
    "float2": ("ff", 2, 2),
    "float3": ("ffff", 4, 3),        # Metal simd_float3 is 16-byte aligned; 4th float is padding
    "packed_float3": ("fff", 3, 3),  # Metal packed_float3, no alignment padding
    "float4": ("ffff", 4, 4),
    "uint32": ("I", 1, 1),
    "int32": ("i", 1, 1),
    "half": ("e", 1, 1),
    "half2": ("ee", 2, 2),
    "half4": ("eeee", 4, 4),
}


def parse_layout(layout: str) -> tuple[str, list[tuple[str, int, int]]]:
    """Parse a layout string into a struct format and field descriptors.

    Returns (struct_format, [(component_name, n_read, n_report), ...]).
    """
    components = [c.strip() for c in layout.split(",")]
    fmt_parts: list[str] = []
    fields: list[tuple[str, int, int]] = []

    for comp in components:
        if comp not in _LAYOUT_FORMATS:
            raise ValueError(f"Unknown layout component: {comp}")
        fmt, n_read, n_report = _LAYOUT_FORMATS[comp]
        fmt_parts.append(fmt)
        fields.append((comp, n_read, n_report))

    return "<" + "".join(fmt_parts), fields


def read_buffer(
    gputrace: Path,
    filename: str,
    layout: str,
    start: int = 0,
    count: int = 10,
) -> list[BufferElement]:
    """Read a buffer file with the given layout.

    Layout formats:
        "float"                  - single float per element
        "float4"                 - 4 floats (RGBA, position, etc.)
        "float4,float4,float4"  - compound struct (e.g., Particle)
        "uint32"                 - single uint32 per element
        "half4"                  - 4 half-precision floats
    """
    filepath = gputrace / filename
    if not filepath.exists():
        return []

    fmt, fields = parse_layout(layout)
    stride = struct.calcsize(fmt)

    data = filepath.read_bytes()
    total_elements = len(data) // stride

    results: list[BufferElement] = []
    for idx in range(start, min(start + count, total_elements)):
        offset = idx * stride
        values = struct.unpack_from(fmt, data, offset)

        field_entries: list[tuple[str, object]] = []
        vi = 0
        for ci, (comp_name, n_read, n_report) in enumerate(fields):
            if n_report == 1:
                field_entries.append((f"field{ci}", values[vi]))
            else:
                field_entries.append(
                    (f"field{ci}", list(values[vi : vi + n_report]))
                )
            vi += n_read
        results.append(BufferElement(index=idx, fields=tuple(field_entries)))

    return results


def buffer_element_count(gputrace: Path, filename: str, layout: str) -> int:
    """Return total element count for a buffer given a layout."""
    filepath = gputrace / filename
    if not filepath.exists():
        return 0
    fmt, _ = parse_layout(layout)
    stride = struct.calcsize(fmt)
    return filepath.stat().st_size // stride


# ── resource listing ──────────────────────────────────────────────────────


def list_resources(gputrace: Path) -> CaptureInfo:
    """List all resources in a .gputrace bundle."""
    labels = extract_buffer_labels(gputrace)
    metadata = parse_metadata(gputrace)
    functions, libraries = extract_shader_info(gputrace)

    resources: list[CaptureResource] = []
    for fname in sorted(os.listdir(gputrace)):
        m = re.match(r"MTL(Buffer|Texture)-\d+-\d+", fname)
        if not m:
            continue
        size = (gputrace / fname).stat().st_size
        kind = "buffer" if m.group(1) == "Buffer" else "texture"
        label = labels.get(fname, "")
        resources.append(
            CaptureResource(
                filename=fname,
                size_bytes=size,
                label=label,
                kind=kind,
            )
        )

    return CaptureInfo(
        metadata=metadata,
        resources=tuple(resources),
        shader_functions=tuple(functions),
        shader_libraries=tuple(ShaderLibrary(path=p) for p in libraries),
    )


# ── buffer stats (no numpy dependency) ───────────────────────────────────


@dataclass(frozen=True)
class BufferStats:
    filename: str
    label: str
    size_bytes: int
    float_count: int
    min_val: float
    max_val: float
    mean_val: float
    first_four: tuple[float, ...]


def buffer_summary(gputrace: Path, filename: str, label: str) -> BufferStats | None:
    """Compute summary statistics for a buffer interpreted as float32."""
    filepath = gputrace / filename
    if not filepath.exists():
        return None

    data = filepath.read_bytes()
    n_floats = len(data) // 4
    if n_floats == 0:
        return None

    values = struct.unpack_from(f"<{n_floats}f", data)

    return BufferStats(
        filename=filename,
        label=label,
        size_bytes=len(data),
        float_count=n_floats,
        min_val=min(values),
        max_val=max(values),
        mean_val=sum(values) / n_floats,
        first_four=tuple(values[:4]),
    )


def resolve_buffer_filename(
    gputrace: Path,
    buffer_query: str,
    labels: dict[str, str] | None = None,
) -> str | None:
    """Resolve a buffer label or filename to an actual filename.

    Tries exact filename match first, then exact label match,
    then partial label match (case-insensitive).
    """
    if (gputrace / buffer_query).exists():
        return buffer_query

    if labels is None:
        labels = extract_buffer_labels(gputrace)

    # Exact label match (case-insensitive)
    for fname, label in labels.items():
        if label.lower() == buffer_query.lower():
            return fname

    # Partial match
    for fname, label in labels.items():
        if buffer_query.lower() in label.lower():
            return fname

    return None
