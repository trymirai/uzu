import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matmul_bench.models import BenchmarkRun

DISPATCH_STYLES: dict[str, dict] = {
    "Gemv": dict(color="#34A853", marker="^", linestyle="-"),
    "Gemm": dict(color="#EA4335", marker="v", linestyle="--"),
    "GemmMpp": dict(color="#4285F4", marker="o", linestyle="-"),
    "GemmMppDirect": dict(color="#7B1FA2", marker="s", linestyle="-."),
}

DEFAULT_STYLE = dict(color="#757575", marker="x", linestyle="-.")


def _style_for(dispatch_path: str) -> dict:
    return DISPATCH_STYLES.get(dispatch_path, DEFAULT_STYLE)


def _group_results(
    run: BenchmarkRun,
) -> dict[str, dict[tuple[int, int], dict[str, dict[int, float]]]]:
    grouped: dict[str, dict[tuple[int, int], dict[str, dict[int, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for r in run.results:
        if r.status != "ok":
            continue
        grouped[r.combo][(r.k, r.n)][r.dispatch_path][r.m] = r.duration_ms
    return grouped


def generate_charts(
    run: BenchmarkRun,
    output_dir: Path | str,
    source_filename: str = "",
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = _group_results(run)
    generated_files: list[Path] = []

    for combo, kn_data in sorted(grouped.items()):
        path = _generate_combo_chart(combo, kn_data, run.device, output_dir, source_filename)
        generated_files.append(path)

    return generated_files


def _generate_combo_chart(
    combo: str,
    kn_data: dict[tuple[int, int], dict[str, dict[int, float]]],
    device: str,
    output_dir: Path,
    source_filename: str,
) -> Path:
    kn_pairs = sorted(kn_data.keys(), key=lambda kn: kn[0] * kn[1])
    num_panels = len(kn_pairs)
    cols = min(4, num_panels)
    rows = math.ceil(num_panels / cols)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(5.5 * cols, 4.0 * rows),
        squeeze=False,
    )

    fig.suptitle(
        f"{combo} Relative Slowdown vs Gemv(bs=1)",
        fontsize=16,
        fontweight="bold",
        y=1.0,
    )
    subtitle_parts = [device]
    if source_filename:
        subtitle_parts.append(source_filename)
    fig.text(0.5, 0.985, " | ".join(subtitle_parts), ha="center", fontsize=9, color="gray")

    for idx, (k, n) in enumerate(kn_pairs):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        path_data = kn_data[(k, n)]
        _draw_panel(ax, k, n, path_data)

    for idx in range(num_panels, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    safe_combo = combo.replace("*", "x").replace("->", "_").replace(" ", "")
    out_path = output_dir / f"matmul_perf_{safe_combo}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def _draw_panel(
    ax: plt.Axes,
    k: int,
    n: int,
    path_data: dict[str, dict[int, float]],
) -> None:
    ax.set_title(f"N={n}, K={k}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Batch size (M)", fontsize=8)
    ax.set_ylabel("Relative slowdown vs Gemv(bs=1)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, which="both")

    gemv_data = path_data.get("Gemv", {})
    baseline_ms = gemv_data.get(1)
    if baseline_ms is None or baseline_ms == 0:
        for dp in sorted(path_data.keys()):
            if 1 in path_data[dp] and path_data[dp][1] > 0:
                baseline_ms = path_data[dp][1]
                break
    if baseline_ms is None or baseline_ms == 0:
        baseline_ms = 1.0

    for dispatch_path in sorted(path_data.keys()):
        m_to_ms = path_data[dispatch_path]
        if not m_to_ms:
            continue

        m_values = sorted(m_to_ms.keys())
        rel_values = [m_to_ms[m] / baseline_ms for m in m_values]

        style = _style_for(dispatch_path)
        ax.plot(
            m_values, rel_values,
            label=dispatch_path,
            markersize=4,
            linewidth=1.5,
            **style,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3g}x"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.axhline(y=1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.legend(fontsize=7, loc="upper left")
