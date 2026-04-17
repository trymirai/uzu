from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelCounters:
    name: str
    total_ns: int
    count: int
    counters: dict[str, float]

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1e6

    @property
    def avg_us(self) -> float:
        return self.total_ns / max(self.count, 1) / 1e3

    def counter(self, name: str) -> float | None:
        return self.counters.get(name)

    @property
    def bottleneck(self) -> str:
        limiter_counters = {
            k: v for k, v in self.counters.items() if "Limiter" in k and v > 0
        }
        if not limiter_counters:
            return "--"
        top = max(limiter_counters, key=limiter_counters.__getitem__)
        return top.replace(" Limiter", "")


@dataclass(frozen=True)
class TraceProfile:
    label: str
    device: str
    duration_ns: int
    kernels: dict[str, KernelCounters]


_ID_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")


def _normalize_kernel_name(raw_name: str) -> str:
    return _ID_SUFFIX.sub("", raw_name).strip()


def load_trace_profile(json_path: Path | str, label: str) -> TraceProfile:
    with open(json_path) as f:
        data = json.load(f)

    device = data.get("metadata", {}).get("device_name", "unknown")
    duration_ns = data.get("metadata", {}).get("duration_ns", 0)

    kernels: dict[str, KernelCounters] = {}

    for entry in data.get("kernel_counter_stats", []):
        raw_name = entry["name"]
        normalized = _normalize_kernel_name(raw_name)
        counters = {name: value for name, value in entry.get("counters", [])}

        kc = KernelCounters(
            name=normalized,
            total_ns=entry["total_ns"],
            count=entry["count"],
            counters=counters,
        )

        if normalized in kernels:
            existing = kernels[normalized]
            merged_counters = dict(existing.counters)
            for k, v in counters.items():
                if k in merged_counters:
                    w1, w2 = existing.total_ns, kc.total_ns
                    total_w = w1 + w2
                    merged_counters[k] = (merged_counters[k] * w1 + v * w2) / total_w if total_w > 0 else 0
                else:
                    merged_counters[k] = v
            kernels[normalized] = KernelCounters(
                name=normalized,
                total_ns=existing.total_ns + kc.total_ns,
                count=existing.count + kc.count,
                counters=merged_counters,
            )
        else:
            kernels[normalized] = kc

    return TraceProfile(
        label=label,
        device=device,
        duration_ns=duration_ns,
        kernels=kernels,
    )


KEY_COUNTERS = [
    "ALU Utilization",
    "ALU Limiter",
    "Buffer Load Utilization",
    "Buffer Load Limiter",
    "Compute Occupancy",
]


@dataclass(frozen=True)
class KernelRow:
    name: str
    entries: tuple[KernelCounters | None, ...]


def build_comparison(profiles: list[TraceProfile]) -> list[KernelRow]:
    all_names: dict[str, int] = {}
    for profile in profiles:
        for kc in sorted(profile.kernels.values(), key=lambda k: -k.total_ns):
            if kc.name not in all_names:
                all_names[kc.name] = 0
            all_names[kc.name] = max(all_names[kc.name], kc.total_ns)

    sorted_names = sorted(all_names, key=lambda n: -all_names[n])

    return [
        KernelRow(
            name=name,
            entries=tuple(p.kernels.get(name) for p in profiles),
        )
        for name in sorted_names
    ]
