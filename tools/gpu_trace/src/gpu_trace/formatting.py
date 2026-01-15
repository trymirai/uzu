from __future__ import annotations

from .models import KernelCounterStats, TraceExport


def _fmt_ns(ns: int) -> str:
    if ns >= 1_000_000_000:
        return f"{ns / 1e9:.2f} s"
    if ns >= 1_000_000:
        return f"{ns / 1e6:.2f} ms"
    if ns >= 1_000:
        return f"{ns / 1e3:.1f} µs"
    return f"{ns} ns"


def _fmt_percent(value: float) -> str:
    return f"{value:.2f}%"


def _bar(value: float, width: int = 20) -> str:
    filled = int(value / 100 * width)
    return "█" * filled + "░" * (width - filled)


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

    name = kcs.name[:50] if len(kcs.name) <= 50 else kcs.name[:47] + "..."
    header = f" {name} ({kcs.count} calls, {_fmt_ns(kcs.total_ns)} total)"

    print("┌" + "─" * 78 + "┐")
    print("│" + header.ljust(78) + "│")
    print("├" + "─" * 78 + "┤")

    alu_lim = counter_dict.get("ALU Limiter", 0)
    alu_util = counter_dict.get("ALU Utilization", 0)
    f32 = counter_dict.get("F32 Utilization", 0)
    f16 = counter_dict.get("F16 Utilization", 0)
    row = f"  ALU:     Limiter {alu_lim:>5.1f}%  Util {alu_util:>5.1f}%  F32 {f32:>5.1f}%  F16 {f16:>5.1f}%"
    print("│" + row.ljust(78) + "│")

    buf_rd_lim = counter_dict.get("Buffer Read Limiter", 0)
    buf_ld = counter_dict.get("Buffer Load Utilization", 0)
    buf_wr_lim = counter_dict.get("Buffer Write Limiter", 0)
    buf_st = counter_dict.get("Buffer Store Utilization", 0)
    row = f"  Buffer:  RdLim {buf_rd_lim:>5.1f}%  Load {buf_ld:>5.1f}%  WrLim {buf_wr_lim:>5.1f}%  Store {buf_st:>5.1f}%"
    print("│" + row.ljust(78) + "│")

    gpu_rd = counter_dict.get("GPU Read Bandwidth", 0)
    gpu_wr = counter_dict.get("GPU Write Bandwidth", 0)
    row = f"  Memory:  Read {gpu_rd:>6.1f} GB/s  Write {gpu_wr:>6.1f} GB/s"
    print("│" + row.ljust(78) + "│")

    llc_lim = counter_dict.get("GPU Last Level Cache Limiter", 0)
    llc_util = counter_dict.get("GPU Last Level Cache Utilization", 0)
    tex_lim = counter_dict.get("Texture Sample Limiter", 0)
    tex_util = counter_dict.get("Texture Sample Utilization", 0)
    row = f"  Cache:   LLC Lim {llc_lim:>5.1f}%  Util {llc_util:>5.1f}%  Tex {tex_lim:>5.1f}%/{tex_util:>5.1f}%"
    print("│" + row.ljust(78) + "│")

    tg_ld_lim = counter_dict.get("Threadgroup/Imageblock Load Limiter", 0)
    tg_ld = counter_dict.get("Threadgroup/Imageblock Load Utilization", 0)
    tg_st_lim = counter_dict.get("Threadgroup/Imageblock Store Limiter", 0)
    tg_st = counter_dict.get("Threadgroup/Imageblock Store Utilization", 0)
    row = f"  TGroup:  LdLim {tg_ld_lim:>5.1f}%  Load {tg_ld:>5.1f}%  StLim {tg_st_lim:>5.1f}%  Store {tg_st:>5.1f}%"
    print("│" + row.ljust(78) + "│")

    compute_occ = counter_dict.get("Compute Occupancy", 0)
    mmu_lim = counter_dict.get("MMU Limiter", 0)
    mmu_util = counter_dict.get("MMU Utilization", 0)
    row = f"  Occup:   Compute {compute_occ:>5.1f}%  MMU Lim {mmu_lim:>5.1f}%  Util {mmu_util:>5.1f}%"
    print("│" + row.ljust(78) + "│")

    bottleneck = _find_bottleneck(counter_dict)
    if bottleneck:
        bn_name, bn_value = bottleneck
        row = f"  >>> Bottleneck: {bn_name} ({bn_value:.1f}%)"
        print("│" + row.ljust(78) + "│")

    print("└" + "─" * 78 + "┘")


def print_summary(export: TraceExport, verbose: bool = False) -> None:
    meta = export.metadata
    util = export.utilization

    print()
    print("┌" + "─" * 78 + "┐")
    print("│" + " GPU TRACE SUMMARY ".center(78) + "│")
    print("├" + "─" * 78 + "┤")

    print("│" + f"  Device: {meta.device_model} ({meta.device_name})".ljust(78) + "│")
    print("│" + f"  OS: {meta.os_version}".ljust(78) + "│")
    print(
        "│"
        + f"  Process: {meta.target_process} (PID {meta.target_pid})".ljust(78)
        + "│"
    )
    print("│" + f"  Template: {meta.template_name}".ljust(78) + "│")
    print("│" + f"  Duration: {_fmt_ns(util.run_duration_ns)}".ljust(78) + "│")

    print("├" + "─" * 78 + "┤")
    print("│" + " UTILIZATION ".center(78) + "│")
    print("├" + "─" * 78 + "┤")

    global_pct = util.global_utilization_percent
    target_pct = util.target_utilization_percent
    window_pct = util.window_utilization_percent

    print(
        "│"
        + f"  Global GPU Active:  {_bar(global_pct)} {_fmt_percent(global_pct):>8}  ({_fmt_ns(util.global_active_ns)})".ljust(
            78
        )
        + "│"
    )
    print(
        "│"
        + f"  Target GPU Active:  {_bar(target_pct)} {_fmt_percent(target_pct):>8}  ({_fmt_ns(util.target_active_ns)})".ljust(
            78
        )
        + "│"
    )
    print(
        "│"
        + f"  Window Utilization: {_bar(window_pct)} {_fmt_percent(window_pct):>8}  (when active)".ljust(
            78
        )
        + "│"
    )
    print("│" + f"  Command Buffers: {util.command_buffer_count}".ljust(78) + "│")

    if export.performance_states:
        states: dict[str, int] = {}
        for ps in export.performance_states:
            key = ps.state.value
            states[key] = states.get(key, 0) + ps.duration_ns
        induced_count = sum(1 for ps in export.performance_states if ps.is_induced)
        print("├" + "─" * 78 + "┤")
        print("│" + " GPU PERFORMANCE STATES ".center(78) + "│")
        print("├" + "─" * 78 + "┤")
        for state_name, duration in sorted(states.items(), key=lambda x: -x[1]):
            print("│" + f"  {state_name}: {_fmt_ns(duration)}".ljust(78) + "│")
        if induced_count:
            print(
                "│" + f"  ⚠ Induced states: {induced_count} intervals".ljust(78) + "│"
            )

    if export.counter_definitions:
        print("├" + "─" * 78 + "┤")
        print("│" + " AVAILABLE COUNTERS ".center(78) + "│")
        print("├" + "─" * 78 + "┤")
        print(
            "│"
            + f"  {len(export.counter_definitions)} GPU hardware counters available".ljust(
                78
            )
            + "│"
        )

    print("├" + "─" * 78 + "┤")
    print("│" + " KERNEL DISPATCHES ".center(78) + "│")
    print("├" + "─" * 78 + "┤")
    print("│" + f"  Total dispatches: {len(export.dispatches):,}".ljust(78) + "│")
    print("│" + f"  Unique kernels: {len(export.kernel_summary)}".ljust(78) + "│")
    print("│" + f"  Shaders compiled: {len(export.shaders)}".ljust(78) + "│")

    if export.kernel_summary:
        print("├" + "─" * 78 + "┤")
        print("│" + " TOP 15 KERNELS BY TIME ".center(78) + "│")
        print("├" + "─" * 78 + "┤")
        header = f"  {'Kernel':<30}  {'Total':>10}  {'Avg':>10}  {'Count':>7}  {'%':>6}"
        print("│" + header.ljust(78) + "│")
        print("├" + "─" * 78 + "┤")

        total_ns = sum(k.total_ns for k in export.kernel_summary)
        for k in export.kernel_summary[:15]:
            pct = (k.total_ns / total_ns * 100) if total_ns else 0
            name = k.name[:30] if len(k.name) <= 30 else k.name[:27] + "..."
            row = f"  {name:<30}  {_fmt_ns(k.total_ns):>10}  {_fmt_ns(int(k.avg_ns)):>10}  {k.count:>7}  {pct:>5.1f}%"
            print("│" + row.ljust(78) + "│")

    if export.kernel_counter_stats:
        if verbose:
            print("├" + "─" * 78 + "┤")
            print("│" + " TOP 10 KERNELS - DETAILED COUNTERS ".center(78) + "│")
            print("└" + "─" * 78 + "┘")
            print()
            for kcs in export.kernel_counter_stats[:10]:
                if kcs.counters:
                    _print_kernel_detail(kcs)
                    print()
            return
        else:
            print("├" + "─" * 78 + "┤")
            print("│" + " TOP 10 KERNELS WITH COUNTERS ".center(78) + "│")
            print("├" + "─" * 78 + "┤")
            header = f"  {'Kernel':<26}  {'ALU':>8}  {'BufLd':>8}  {'BufSt':>8}  {'Occup':>8}  {'Cache':>8}"
            print("│" + header.ljust(78) + "│")
            print("├" + "─" * 78 + "┤")

            for kcs in export.kernel_counter_stats[:10]:
                if not kcs.counters:
                    continue
                counter_dict = dict(kcs.counters)
                alu = counter_dict.get("ALU Utilization", 0)
                buf_ld = counter_dict.get("Buffer Load Utilization", 0)
                buf_st = counter_dict.get("Buffer Store Utilization", 0)
                occup = counter_dict.get("Compute Occupancy", 0)
                cache = counter_dict.get("GPU Last Level Cache Utilization", 0)

                name = kcs.name[:26] if len(kcs.name) <= 26 else kcs.name[:23] + "..."
                row = f"  {name:<26}  {alu:>7.1f}%  {buf_ld:>7.1f}%  {buf_st:>7.1f}%  {occup:>7.1f}%  {cache:>7.1f}%"
                print("│" + row.ljust(78) + "│")

            print("└" + "─" * 78 + "┘")
            print()
            return

    print("└" + "─" * 78 + "┘")
    print()


def print_comparison(
    export1: TraceExport,
    export2: TraceExport,
    label1: str,
    label2: str,
) -> None:
    print()
    print("┌" + "─" * 78 + "┐")
    print("│" + " GPU TRACE COMPARISON ".center(78) + "│")
    print("├" + "─" * 78 + "┤")

    print("│" + f"  {label1}: {export1.metadata.target_process}".ljust(78) + "│")
    print("│" + f"  {label2}: {export2.metadata.target_process}".ljust(78) + "│")

    print("├" + "─" * 78 + "┤")
    print("│" + " UTILIZATION ".center(78) + "│")
    print("├" + "─" * 78 + "┤")

    header = f"  {'Metric':<22}  {label1:>14}  {label2:>14}  {'Diff':>14}"
    print("│" + header.ljust(78) + "│")
    print("├" + "─" * 78 + "┤")

    def _diff_str(v1: float, v2: float) -> str:
        if v1 == 0:
            return "n/a"
        diff = (v2 - v1) / v1 * 100
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.1f}%"

    u1, u2 = export1.utilization, export2.utilization

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

    for metric, v1, v2, diff in rows:
        row = f"  {metric:<22}  {v1:>14}  {v2:>14}  {diff:>14}"
        print("│" + row.ljust(78) + "│")

    print("├" + "─" * 78 + "┤")
    print("│" + " TOP KERNELS COMPARISON ".center(78) + "│")
    print("├" + "─" * 78 + "┤")

    header = f"  {'Kernel':<32}  {label1:>12}  {label2:>12}  {'Diff':>10}"
    print("│" + header.ljust(78) + "│")
    print("├" + "─" * 78 + "┤")

    kernel_map1 = {k.name: k for k in export1.kernel_summary}
    kernel_map2 = {k.name: k for k in export2.kernel_summary}
    all_names = set(kernel_map1.keys()) | set(kernel_map2.keys())

    combined: list[tuple[str, int, int, str]] = []
    for name in all_names:
        k1 = kernel_map1.get(name)
        k2 = kernel_map2.get(name)
        ns1 = k1.total_ns if k1 else 0
        ns2 = k2.total_ns if k2 else 0
        diff = _diff_str(ns1, ns2)
        combined.append((name, ns1, ns2, diff))

    combined.sort(key=lambda x: x[1] + x[2], reverse=True)

    for name, ns1, ns2, diff in combined[:15]:
        short_name = name[:32] if len(name) <= 32 else name[:29] + "..."
        row = f"  {short_name:<32}  {_fmt_ns(ns1):>12}  {_fmt_ns(ns2):>12}  {diff:>10}"
        print("│" + row.ljust(78) + "│")

    total1 = sum(k.total_ns for k in export1.kernel_summary)
    total2 = sum(k.total_ns for k in export2.kernel_summary)

    print("├" + "─" * 78 + "┤")
    row = f"  {'TOTAL':<32}  {_fmt_ns(total1):>12}  {_fmt_ns(total2):>12}  {_diff_str(total1, total2):>10}"
    print("│" + row.ljust(78) + "│")

    print("└" + "─" * 78 + "┘")
    print()
