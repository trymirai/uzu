#!/usr/bin/env python3
"""Render an interleaved knob sweep (tiling or split-k target) as per-shape tables.

Usage: a8w_knob_report.py <family> <label>[=<path.json>]   family: A8WTile | A8WSplitK

Every candidate for one (m, method) is measured back to back in one process, bracketed
by two `default` readings. Each candidate is scored against the mean of that bracket, so
a transient that hits the whole block cancels; the bracket's own spread is reported as
the error bar for that row.
"""

import json
import sys

MS = [16, 32, 64, 128, 256, 512, 1024, 2048]
SHAPES = ["k2048_n3072", "k1024_n7168", "k2048_n12288", "k3584_n1024", "k9216_n2560"]
METHODS = ["sym", "bias", "zp"]


def main() -> None:
    family = sys.argv[1]
    label, _, path = sys.argv[2].partition("=")
    rows = json.load(open(path or f"bench_results/{label}.json"))

    for shape in SHAPES:
        prefix = f"Metal_{family}_w8_{shape}/"
        cells = {}
        canary = []
        for key, value in rows.items():
            if not key.startswith(prefix):
                continue
            _, name, parameter = key.split("/")
            if "canary" in name:
                canary.append(value)
            else:
                cells[(name, parameter)] = value
        if not cells:
            continue

        names = sorted({n for (n, _) in cells if not n.startswith("default")})
        spread = (max(canary) - min(canary)) / min(canary) * 100 if len(canary) >= 2 else float("nan")
        print(f"\n### {shape}   (canary spread {spread:.2f}%)\n")

        for method in METHODS:
            candidates = [n for n in names if n.endswith(f"_{method}")]
            if not candidates:
                continue
            print(f"**{method}** - candidate / bracketing default (<1.000 = faster than today)\n")
            header = [n[: -len(method) - 1] for n in candidates]
            print("| m | " + " | ".join(header) + " | best | gain | bracket |")
            print("|--:|" + "--:|" * (len(header) + 3))
            for m in MS:
                parameter = f"m{m}"
                first = cells.get((f"default_{method}", parameter))
                second = cells.get((f"default2_{method}", parameter))
                if first is None or second is None:
                    continue
                base = (first + second) / 2.0
                bracket = abs(second - first) / base * 100.0
                best_name, best_ratio, out = None, None, []
                for name, label_ in zip(candidates, header):
                    value = cells.get((name, parameter))
                    if value is None:
                        out.append("-")
                        continue
                    ratio = value / base
                    out.append(f"{ratio:.3f}")
                    if best_ratio is None or ratio < best_ratio:
                        best_name, best_ratio = label_, ratio
                if best_ratio is None:
                    continue
                marked = [f"**{c}**" if h == best_name else c for h, c in zip(header, out)]
                print(
                    f"| {m} | " + " | ".join(marked)
                    + f" | {best_name} | {(best_ratio - 1) * 100:+.1f}% | {bracket:.1f}% |"
                )
            print()


if __name__ == "__main__":
    main()
