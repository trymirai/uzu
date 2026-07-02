from __future__ import annotations

import csv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .models import GpuPerformanceStateInterval, KernelDispatch, TraceExport


def write_dispatch_csv(export: TraceExport, output: Path) -> None:
    counter_names = sorted(
        {counter_name for stats in export.kernel_counter_stats for counter_name, _ in stats.counters},
    )
    fieldnames = [
        "kernel_name",
        "start_ns",
        "duration_ns",
        "end_ns",
        "command_buffer_id",
        "channel",
        "previous_kernel_name",
        "next_kernel_name",
        "performance_state",
        "performance_state_is_induced",
        "device_model",
        "os_version",
        "target_process",
        "template_name",
        *[f"counter_{name}" for name in counter_names],
    ]

    counter_map = {stats.name: dict(stats.counters) for stats in export.kernel_counter_stats}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, dispatch in enumerate(export.dispatches):
            row = _dispatch_row(
                export,
                dispatch,
                previous_dispatch=export.dispatches[index - 1] if index > 0 else None,
                next_dispatch=export.dispatches[index + 1] if index + 1 < len(export.dispatches) else None,
                performance_state=_performance_state_at(export.performance_states, dispatch.start_ns),
            )
            counters = counter_map.get(dispatch.name, {})
            for counter_name in counter_names:
                row[f"counter_{counter_name}"] = counters.get(counter_name, "")
            writer.writerow(row)


def _dispatch_row(
    export: TraceExport,
    dispatch: KernelDispatch,
    previous_dispatch: KernelDispatch | None,
    next_dispatch: KernelDispatch | None,
    performance_state: GpuPerformanceStateInterval | None,
) -> dict[str, object]:
    return {
        "kernel_name": dispatch.name,
        "start_ns": dispatch.start_ns,
        "duration_ns": dispatch.duration_ns,
        "end_ns": dispatch.start_ns + dispatch.duration_ns,
        "command_buffer_id": dispatch.command_buffer_id,
        "channel": dispatch.channel,
        "previous_kernel_name": previous_dispatch.name if previous_dispatch is not None else "",
        "next_kernel_name": next_dispatch.name if next_dispatch is not None else "",
        "performance_state": performance_state.state.value if performance_state is not None else "",
        "performance_state_is_induced": performance_state.is_induced if performance_state is not None else "",
        "device_model": export.metadata.device_model,
        "os_version": export.metadata.os_version,
        "target_process": export.metadata.target_process,
        "template_name": export.metadata.template_name,
    }


def _performance_state_at(
    states: tuple[GpuPerformanceStateInterval, ...],
    timestamp_ns: int,
) -> GpuPerformanceStateInterval | None:
    for state in states:
        if state.start_ns <= timestamp_ns < state.start_ns + state.duration_ns:
            return state
    return None
