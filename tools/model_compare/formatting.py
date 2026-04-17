from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .comparison import KEY_COUNTERS, KernelCounters, KernelRow, TraceProfile


def _fmt_time(kc: KernelCounters | None) -> str:
    if kc is None:
        return "--"
    if kc.total_ms >= 1.0:
        return f"{kc.total_ms:.2f} ms"
    return f"{kc.total_ns / 1e3:.1f} us"


def _fmt_count(kc: KernelCounters | None) -> str:
    if kc is None:
        return "--"
    return str(kc.count)


def _fmt_counter(kc: KernelCounters | None, counter_name: str) -> str:
    if kc is None:
        return "--"
    val = kc.counter(counter_name)
    if val is None:
        return "--"
    return f"{val:.1f}%"


def _fmt_bottleneck(kc: KernelCounters | None) -> str:
    if kc is None:
        return "--"
    return kc.bottleneck


def _short_name(name: str) -> str:
    name = name.removeprefix("_D")
    if len(name) > 1 and name[0].isdigit():
        i = 0
        while i < len(name) and name[i].isdigit():
            i += 1
        name = name[i:]
    if len(name) > 40:
        name = name[:37] + "..."
    return name


def print_comparison_table(
    profiles: list[TraceProfile],
    rows: list[KernelRow],
    console: Console | None = None,
    top_n: int = 15,
) -> None:
    console = console or Console()

    labels = [p.label for p in profiles]
    title = " vs ".join(labels)

    table = Table(title=f"Kernel Comparison: {title}", show_lines=True, expand=False)
    table.add_column("Kernel", style="bold", max_width=30)

    for label in labels:
        table.add_column(f"{label} time", justify="right", max_width=10)
        table.add_column("#", justify="right", max_width=5)

    table.add_column("ALU%", justify="right", max_width=6)
    table.add_column("BufRd%", justify="right", max_width=6)
    table.add_column("Occup%", justify="right", max_width=6)
    table.add_column("Bottleneck", justify="right", max_width=12)

    for row in rows[:top_n]:
        cells: list[str] = [_short_name(row.name)]

        for entry in row.entries:
            cells.append(_fmt_time(entry))
            cells.append(_fmt_count(entry))

        first_present = next((e for e in row.entries if e is not None), None)
        cells.append(_fmt_counter(first_present, "ALU Utilization"))
        cells.append(_fmt_counter(first_present, "Buffer Load Utilization"))
        cells.append(_fmt_counter(first_present, "Compute Occupancy"))
        cells.append(_fmt_bottleneck(first_present))

        table.add_row(*cells)

    total_row: list[str] = ["TOTAL"]
    for profile in profiles:
        total_ns = sum(kc.total_ns for kc in profile.kernels.values())
        total_calls = sum(kc.count for kc in profile.kernels.values())
        total_row.append(f"{total_ns / 1e6:.1f} ms")
        total_row.append(str(total_calls))
    total_row.extend(["", "", "", ""])
    table.add_row(*total_row, style="bold")

    console.print(table)

    console.print()
    for profile in profiles:
        console.print(f"[cyan]{profile.label}[/cyan]: {profile.device}, {profile.duration_ns / 1e9:.2f}s recording")


def print_counter_detail(
    profiles: list[TraceProfile],
    rows: list[KernelRow],
    console: Console | None = None,
    top_n: int = 10,
) -> None:
    console = console or Console()

    for row in rows[:top_n]:
        console.print(f"\n[bold]{_short_name(row.name)}[/bold]")
        detail_table = Table(show_header=True, show_lines=False, pad_edge=False)
        detail_table.add_column("Counter", no_wrap=True)

        present_indices: list[int] = []
        for i, entry in enumerate(row.entries):
            if entry is not None:
                detail_table.add_column(profiles[i].label, justify="right")
                present_indices.append(i)

        if not present_indices:
            continue

        all_counter_names: dict[str, None] = {}
        for idx in present_indices:
            entry = row.entries[idx]
            assert entry is not None
            for k in entry.counters:
                all_counter_names[k] = None

        for counter_name in all_counter_names:
            cells = [counter_name]
            for idx in present_indices:
                entry = row.entries[idx]
                assert entry is not None
                val = entry.counter(counter_name)
                cells.append(f"{val:.2f}" if val is not None else "--")
            detail_table.add_row(*cells)

        timing_cells = ["Total time"]
        for idx in present_indices:
            entry = row.entries[idx]
            assert entry is not None
            timing_cells.append(_fmt_time(entry))
        detail_table.add_row(*timing_cells, style="bold")

        count_cells = ["Dispatches"]
        for idx in present_indices:
            entry = row.entries[idx]
            assert entry is not None
            count_cells.append(str(entry.count))
        detail_table.add_row(*count_cells)

        console.print(detail_table)
