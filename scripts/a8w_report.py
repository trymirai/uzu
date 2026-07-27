#!/usr/bin/env python3
"""Render a8w bench snapshots as markdown tables.

Usage: a8w_report.py <label>[=<path.json>] [<label>[=<path.json>] ...]

With one snapshot, prints absolute times plus speedup and correction-cost ratios.
With two or more, adds a per-cell comparison against the first (the baseline).
"""

import json
import sys

MS = [16, 32, 64, 128, 256, 512, 1024, 2048]
SHAPES = ["k2048_n3072", "k1024_n7168", "k2048_n12288", "k3584_n1024", "k9216_n2560"]


def load(spec):
    label, _, path = spec.partition("=")
    return label, json.load(open(path or f"bench_results/{label}.json"))


def group(shape):
    return f"Metal_A8W_w8_{shape}"


def get(rows, shape, variant, m):
    return rows.get(f"{group(shape)}/{variant}/m{m}")


def canary_spread(rows, shape):
    values = [x for key, x in rows.items() if key.startswith(group(shape)) and "canary" in key]
    if len(values) < 2:
        return None
    return (max(values) - min(values)) / min(values) * 100.0


def a8_sym(rows, shape, m):
    """Mean of the bracketing A/B/A sym readings, which is the drift-corrected estimate."""
    first, second = get(rows, shape, "a8_gemm_sym", m), get(rows, shape, "a8_gemm_sym2", m)
    if first is None:
        return None
    return (first + second) / 2.0 if second is not None else first


def aba_drift(rows, shape, m):
    first, second = get(rows, shape, "a8_gemm_sym", m), get(rows, shape, "a8_gemm_sym2", m)
    if first is None or second is None:
        return None
    return (second - first) / first * 100.0


def emit_shape(snapshots, shape):
    if all(a8_sym(rows, shape, MS[0]) is None for _, rows in snapshots):
        return

    bar = ", ".join(
        f"{label} {canary_spread(rows, shape):.2f}%"
        for label, rows in snapshots
        if canary_spread(rows, shape) is not None
    )
    print(f"\n### {shape}\n")
    print(f"Canary spread: {bar}\n")

    label, rows = snapshots[-1]
    print(f"**{label} — absolute (µs, GEMM only) and speedup vs a16w8**\n")
    print("| m | a8 sym | a8 bias | a8 zp | bf sym | bf bias | bf zp | sp sym | sp bias | sp zp | bias/sym | zp/sym | A/B/A |")
    print("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for m in MS:
        sym = a8_sym(rows, shape, m)
        if sym is None:
            continue
        bias, zp = get(rows, shape, "a8_gemm_bias", m), get(rows, shape, "a8_gemm_zp", m)
        bf_sym = get(rows, shape, "bf16_gemm_sym", m)
        bf_bias, bf_zp = get(rows, shape, "bf16_gemm_bias", m), get(rows, shape, "bf16_gemm_zp", m)
        drift = aba_drift(rows, shape, m)
        print(
            f"| {m} | {sym:.2f} | {bias:.2f} | {zp:.2f} | {bf_sym:.2f} | {bf_bias:.2f} | {bf_zp:.2f} "
            f"| {bf_sym / sym:.2f} | {bf_bias / bias:.2f} | {bf_zp / zp:.2f} "
            f"| {bias / sym:.2f} | {zp / sym:.2f} | {drift:+.2f}% |"
        )

    if len(snapshots) < 2:
        return
    base_label, base = snapshots[0]
    print(f"\n**Change vs {base_label} (negative = faster)**\n")
    print("| m | " + " | ".join(f"{v}" for v in ("sym", "bias", "zp")) + " |")
    print("|--:|--:|--:|--:|")
    for m in MS:
        cells = []
        for variant in ("a8_gemm_sym", "a8_gemm_bias", "a8_gemm_zp"):
            before = a8_sym(base, shape, m) if variant == "a8_gemm_sym" else get(base, shape, variant, m)
            after = a8_sym(rows, shape, m) if variant == "a8_gemm_sym" else get(rows, shape, variant, m)
            cells.append("-" if before is None or after is None else f"{(after - before) / before * 100:+.1f}%")
        if all(c == "-" for c in cells):
            continue
        print(f"| {m} | " + " | ".join(cells) + " |")


def main():
    snapshots = [load(spec) for spec in sys.argv[1:]]
    for shape in SHAPES:
        emit_shape(snapshots, shape)


if __name__ == "__main__":
    main()
