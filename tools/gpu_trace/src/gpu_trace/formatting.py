from __future__ import annotations

import shutil

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import KernelCounterStats, TraceExport

# Use terminal width or sensible default (minimum 140 for readable tables)
_term_width = shutil.get_terminal_size((140, 24)).columns
console = Console(width=max(_term_width, 140))


def _fmt_ns(ns: int) -> str:
    if ns >= 1_000_000_000:
        return f"{ns / 1e9:.2f} s"
    if ns >= 1_000_000:
        return f"{ns / 1e6:.2f} ms"
    if ns >= 1_000:
        return f"{ns / 1e3:.1f} µs"
    return f"{ns} ns"


def _fmt_percent(value: float) -> str:
    return f"{value:.1f}%"


def _bar(value: float, width: int = 20) -> str:
    filled = int(value / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _color_percent(value: float) -> tuple[str, str]:
    if value >= 80:
        return f"{value:.1f}%", "green"
    if value >= 50:
        return f"{value:.1f}%", "yellow"
    if value >= 20:
        return f"{value:.1f}%", "dim"
    return f"{value:.1f}%", "red"


def _find_bottleneck(counter_dict: dict[str, float]) -> tuple[str, float] | None:
    limiters = [
        ("ALU Limiter", "ALU"),
        ("Buffer Read Limiter", "Buffer Read"),
        ("Buffer Write Limiter", "Buffer Write"),
        ("Texture Sample Limiter", "Texture Sample"),
        ("Texture Filtering Limiter", "Texture Filter"),
        ("GPU Last Level Cache Limiter", "LLC"),
        ("Threadgroup/Imageblock Load Limiter", "TG Load"),
        ("Threadgroup/Imageblock Store Limiter", "TG Store"),
        ("MMU Limiter", "MMU"),
    ]
    max_limiter = None
    max_value = 0.0
    for counter_name, short_name in limiters:
        value = counter_dict.get(counter_name, 0.0)
        if value > max_value:
            max_value = value
            max_limiter = short_name
    if max_limiter and max_value > 0:
        return (max_limiter, max_value)
    return None


def _print_kernel_detail(kcs: KernelCounterStats) -> None:
    counter_dict = dict(kcs.counters)

    table = Table(
        title=f"[bold cyan]{kcs.name}[/]",
        caption=f"{kcs.count}× calls = {_fmt_ns(kcs.total_ns)} total",
        box=box.ROUNDED,
        show_header=False,
        padding=(0, 1),
    )
    table.add_column("Category", style="yellow", width=12)
    table.add_column("Details", min_width=60)

    alu_lim = counter_dict.get("ALU Limiter", 0)
    alu_util = counter_dict.get("ALU Utilization", 0)
    f32 = counter_dict.get("F32 Utilization", 0)
    f16 = counter_dict.get("F16 Utilization", 0)
    table.add_row(
        "ALU",
        f"Util [green]{alu_util:>5.1f}%[/]  Limiter {alu_lim:>5.1f}%  F32 {f32:>5.1f}%  F16 {f16:>5.1f}%",
    )

    buf_ld = counter_dict.get("Buffer Load Utilization", 0)
    buf_rd_lim = counter_dict.get("Buffer Read Limiter", 0)
    buf_st = counter_dict.get("Buffer Store Utilization", 0)
    buf_wr_lim = counter_dict.get("Buffer Write Limiter", 0)
    table.add_row(
        "Buffer",
        f"Load {buf_ld:>5.1f}% (lim {buf_rd_lim:>5.1f}%)  Store {buf_st:>5.1f}% (lim {buf_wr_lim:>5.1f}%)",
    )

    gpu_rd = counter_dict.get("GPU Read Bandwidth", 0)
    gpu_wr = counter_dict.get("GPU Write Bandwidth", 0)
    table.add_row(
        "Memory",
        f"Read [cyan]{gpu_rd:>6.1f}[/] GB/s  Write [cyan]{gpu_wr:>6.1f}[/] GB/s",
    )

    llc_util = counter_dict.get("GPU Last Level Cache Utilization", 0)
    llc_lim = counter_dict.get("GPU Last Level Cache Limiter", 0)
    tex_util = counter_dict.get("Texture Sample Utilization", 0)
    tex_lim = counter_dict.get("Texture Sample Limiter", 0)
    table.add_row(
        "Cache",
        f"LLC {llc_util:>5.1f}% (lim {llc_lim:>5.1f}%)  Texture {tex_util:>5.1f}% (lim {tex_lim:>5.1f}%)",
    )

    tg_ld = counter_dict.get("Threadgroup/Imageblock Load Utilization", 0)
    tg_ld_lim = counter_dict.get("Threadgroup/Imageblock Load Limiter", 0)
    tg_st = counter_dict.get("Threadgroup/Imageblock Store Utilization", 0)
    tg_st_lim = counter_dict.get("Threadgroup/Imageblock Store Limiter", 0)
    table.add_row(
        "Threadgroup",
        f"Load {tg_ld:>5.1f}% (lim {tg_ld_lim:>5.1f}%)  Store {tg_st:>5.1f}% (lim {tg_st_lim:>5.1f}%)",
    )

    compute_occ = counter_dict.get("Compute Occupancy", 0)
    mmu_util = counter_dict.get("MMU Utilization", 0)
    mmu_lim = counter_dict.get("MMU Limiter", 0)
    table.add_row(
        "Occupancy",
        f"Compute [green]{compute_occ:>5.1f}%[/]  MMU {mmu_util:>5.1f}% (lim {mmu_lim:>5.1f}%)",
    )

    bottleneck = _find_bottleneck(counter_dict)
    if bottleneck:
        bn_name, bn_value = bottleneck
        table.add_row(
            "[red]⚠ Bottleneck[/]", f"[red bold]{bn_name} ({bn_value:.1f}%)[/]"
        )

    console.print(table)


def print_summary(export: TraceExport, verbose: bool = False) -> None:
    meta = export.metadata
    util = export.utilization

    # Device info
    device_table = Table(title="[bold]GPU Trace Summary[/]", box=box.ROUNDED)
    device_table.add_column("Property", style="dim")
    device_table.add_column("Value", style="green")
    device_table.add_row("Device", f"{meta.device_model} ({meta.device_name})")
    device_table.add_row("OS", meta.os_version)
    device_table.add_row("Process", f"{meta.target_process} (PID {meta.target_pid})")
    device_table.add_row("Template", meta.template_name)
    device_table.add_row("Duration", _fmt_ns(util.run_duration_ns))
    console.print(device_table)
    console.print()

    # Utilization
    util_table = Table(title="[bold green]◉ Utilization[/]", box=box.ROUNDED)
    util_table.add_column("Metric", style="dim")
    util_table.add_column("Bar", min_width=22)
    util_table.add_column("Value", justify="right")
    util_table.add_column("Time", style="dim")

    global_pct = util.global_utilization_percent
    target_pct = util.target_utilization_percent
    window_pct = util.window_utilization_percent

    g_val, g_color = _color_percent(global_pct)
    t_val, t_color = _color_percent(target_pct)
    w_val, w_color = _color_percent(window_pct)

    util_table.add_row(
        "Global GPU Active",
        _bar(global_pct),
        f"[{g_color}]{g_val}[/]",
        _fmt_ns(util.global_active_ns),
    )
    util_table.add_row(
        "Target GPU Active",
        _bar(target_pct),
        f"[{t_color}]{t_val}[/]",
        _fmt_ns(util.target_active_ns),
    )
    util_table.add_row(
        "Window Util (peak)", _bar(window_pct), f"[{w_color}]{w_val}[/]", "when active"
    )
    util_table.add_row(
        "Command Buffers", "", f"[cyan]{util.command_buffer_count}[/]", ""
    )
    console.print(util_table)
    console.print()

    # Performance states
    if export.performance_states:
        states: dict[str, int] = {}
        for ps in export.performance_states:
            key = ps.state.value
            states[key] = states.get(key, 0) + ps.duration_ns
        induced_count = sum(1 for ps in export.performance_states if ps.is_induced)

        ps_table = Table(title="[bold]GPU Performance States[/]", box=box.ROUNDED)
        ps_table.add_column("State", style="yellow")
        ps_table.add_column("Duration", justify="right", style="green")
        for state_name, duration in sorted(states.items(), key=lambda x: -x[1]):
            ps_table.add_row(state_name, _fmt_ns(duration))
        if induced_count:
            ps_table.add_row(
                "[dim]⚠ Induced states[/]", f"[dim]{induced_count} intervals[/]"
            )
        console.print(ps_table)
        console.print()

    # Counters available
    if export.counter_definitions:
        console.print(
            f"[dim]{len(export.counter_definitions)} GPU hardware counters available[/]"
        )
        console.print()

    # Dispatch summary
    console.print(
        f"[bold]Kernel Dispatches:[/] {len(export.dispatches):,} total, {len(export.kernel_summary)} unique, {len(export.shaders)} shaders compiled"
    )
    console.print()

    # Top kernels by time
    if export.kernel_summary:
        kernel_table = Table(
            title="[bold cyan]⚡ Top 15 Kernels by Time[/]",
            box=box.ROUNDED,
            expand=False,
        )
        kernel_table.add_column(
            "Kernel",
            style="white",
            min_width=40,
            max_width=50,
            no_wrap=True,
            overflow="ellipsis",
        )
        kernel_table.add_column(
            "Total", justify="right", style="green", no_wrap=True, min_width=10
        )
        kernel_table.add_column("Share", no_wrap=True, min_width=18)
        kernel_table.add_column(
            "Count", justify="right", style="dim", no_wrap=True, min_width=6
        )
        kernel_table.add_column(
            "Avg", justify="right", style="cyan", no_wrap=True, min_width=10
        )

        total_ns = sum(k.total_ns for k in export.kernel_summary)
        for k in export.kernel_summary[:15]:
            pct = (k.total_ns / total_ns * 100) if total_ns else 0
            bar = _bar(pct * 2, width=8)  # scale up for visibility
            kernel_table.add_row(
                k.name,
                _fmt_ns(k.total_ns),
                f"{bar} {pct:>5.1f}%",
                str(k.count),
                _fmt_ns(int(k.avg_ns)),
            )
        console.print(kernel_table)
        console.print()

    # Kernel counter stats
    if export.kernel_counter_stats:
        if verbose:
            console.print(
                Panel(
                    "[bold magenta]📊 Top 10 Kernels - Detailed HW Counters[/]",
                    box=box.ROUNDED,
                )
            )
            console.print()
            for kcs in export.kernel_counter_stats[:10]:
                if kcs.counters:
                    _print_kernel_detail(kcs)
                    console.print()
        else:
            counter_table = Table(
                title="[bold magenta]📊 Top 10 Kernels - HW Counters[/]",
                box=box.ROUNDED,
                expand=False,
            )
            counter_table.add_column(
                "Kernel",
                style="white",
                min_width=40,
                max_width=50,
                no_wrap=True,
                overflow="ellipsis",
            )
            counter_table.add_column("ALU", justify="right", no_wrap=True, min_width=8)
            counter_table.add_column(
                "BufLoad", justify="right", no_wrap=True, min_width=9
            )
            counter_table.add_column(
                "BufStore", justify="right", no_wrap=True, min_width=9
            )
            counter_table.add_column(
                "Occupancy", justify="right", no_wrap=True, min_width=10
            )
            counter_table.add_column("LLC", justify="right", no_wrap=True, min_width=7)

            for kcs in export.kernel_counter_stats[:10]:
                if not kcs.counters:
                    continue
                counter_dict = dict(kcs.counters)
                alu = counter_dict.get("ALU Utilization", 0)
                buf_ld = counter_dict.get("Buffer Load Utilization", 0)
                buf_st = counter_dict.get("Buffer Store Utilization", 0)
                occup = counter_dict.get("Compute Occupancy", 0)
                cache = counter_dict.get("GPU Last Level Cache Utilization", 0)

                a_val, a_col = _color_percent(alu)
                bl_val, bl_col = _color_percent(buf_ld)
                bs_val, bs_col = _color_percent(buf_st)
                o_val, o_col = _color_percent(occup)
                c_val, c_col = _color_percent(cache)

                counter_table.add_row(
                    kcs.name,
                    f"[{a_col}]{a_val}[/]",
                    f"[{bl_col}]{bl_val}[/]",
                    f"[{bs_col}]{bs_val}[/]",
                    f"[{o_col}]{o_val}[/]",
                    f"[{c_col}]{c_val}[/]",
                )
            console.print(counter_table)
            console.print()


def print_comparison(
    export1: TraceExport,
    export2: TraceExport,
    label1: str,
    label2: str,
) -> None:
    console.print()
    console.print(
        Panel(f"[bold]GPU Trace Comparison[/]\n{label1} vs {label2}", box=box.ROUNDED)
    )
    console.print()

    def _diff_str(v1: float, v2: float) -> tuple[str, str]:
        if v1 == 0:
            return "n/a", "dim"
        diff = (v2 - v1) / v1 * 100
        sign = "+" if diff >= 0 else ""
        color = "red" if diff > 0 else "green" if diff < 0 else "dim"
        return f"{sign}{diff:.1f}%", color

    u1, u2 = export1.utilization, export2.utilization

    # Utilization comparison
    util_table = Table(title="[bold]Utilization Comparison[/]", box=box.ROUNDED)
    util_table.add_column("Metric", style="dim")
    util_table.add_column(label1, justify="right", style="cyan")
    util_table.add_column(label2, justify="right", style="cyan")
    util_table.add_column("Diff", justify="right")

    rows = [
        (
            "Duration",
            _fmt_ns(u1.run_duration_ns),
            _fmt_ns(u2.run_duration_ns),
            _diff_str(u1.run_duration_ns, u2.run_duration_ns),
        ),
        (
            "GPU Active",
            _fmt_ns(u1.target_active_ns),
            _fmt_ns(u2.target_active_ns),
            _diff_str(u1.target_active_ns, u2.target_active_ns),
        ),
        (
            "Window Util",
            _fmt_percent(u1.window_utilization_percent),
            _fmt_percent(u2.window_utilization_percent),
            _diff_str(u1.window_utilization_percent, u2.window_utilization_percent),
        ),
    ]

    for metric, v1, v2, (diff, color) in rows:
        util_table.add_row(metric, v1, v2, f"[{color}]{diff}[/]")
    console.print(util_table)
    console.print()

    # Kernel comparison
    kernel_table = Table(title="[bold]Top Kernels Comparison[/]", box=box.ROUNDED)
    kernel_table.add_column(
        "Kernel", style="white", max_width=40, no_wrap=True, overflow="ellipsis"
    )
    kernel_table.add_column(label1, justify="right", style="cyan")
    kernel_table.add_column(label2, justify="right", style="cyan")
    kernel_table.add_column("Diff", justify="right")

    kernel_map1 = {k.name: k for k in export1.kernel_summary}
    kernel_map2 = {k.name: k for k in export2.kernel_summary}
    all_names = set(kernel_map1.keys()) | set(kernel_map2.keys())

    combined: list[tuple[str, int, int, tuple[str, str]]] = []
    for name in all_names:
        k1 = kernel_map1.get(name)
        k2 = kernel_map2.get(name)
        ns1 = k1.total_ns if k1 else 0
        ns2 = k2.total_ns if k2 else 0
        diff = _diff_str(ns1, ns2)
        combined.append((name, ns1, ns2, diff))

    combined.sort(key=lambda x: x[1] + x[2], reverse=True)

    for name, ns1, ns2, (diff, color) in combined[:15]:
        kernel_table.add_row(name, _fmt_ns(ns1), _fmt_ns(ns2), f"[{color}]{diff}[/]")

    total1 = sum(k.total_ns for k in export1.kernel_summary)
    total2 = sum(k.total_ns for k in export2.kernel_summary)
    total_diff, total_color = _diff_str(total1, total2)
    kernel_table.add_row(
        "[bold]TOTAL[/]",
        f"[bold]{_fmt_ns(total1)}[/]",
        f"[bold]{_fmt_ns(total2)}[/]",
        f"[bold {total_color}]{total_diff}[/]",
    )

    console.print(kernel_table)
    console.print()
