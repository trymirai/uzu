#!/usr/bin/env python3
"""A8W8 vs Abf16W8, full pipeline, per shape and weight scheme.

Usage: a8w_decision.py <label>[=<path.json>]

This is the comparison that drives the ship decision: the model is already W8, so the
question is whether to *also* quantize activations. Both sides run the input Hadamard
(A8 has it fused into the quantize kernel, A16 runs it standalone), so it cancels.

Every number is the full pipeline - activation preparation plus GEMM. The A8 preparation
is dynamic: the activation scale is computed at runtime on every forward pass, per 32
k-elements, nothing calibrated offline.

Weight schemes, all uint8 with a bf16 scale per 64 k-elements:
  sym   symmetric,  W = scale * (W_u8 - 128)
  bias  affine,     W = scale * W_u8 + bias          (MLX-style)
  zp    asymmetric, W = scale * (W_u8 - zero_point)
Only sym avoids the row-sum correction inside the GEMM.
"""

import json
import sys

SHAPE_NK = {
    "k2048_n3072": (3072, 2048),
    "k1024_n7168": (7168, 1024),
    "k2048_n12288": (12288, 2048),
    "k3584_n1024": (1024, 3584),
    "k9216_n2560": (2560, 9216),
}
MS = [16, 32, 64, 128, 256, 512, 1024, 2048]
SCHEMES = [("sym", "symmetric"), ("bias", "affine (scale+bias)"), ("zp", "asymmetric (scale+zero-point)")]


def main() -> None:
    label, _, path = sys.argv[1].partition("=")
    rows = json.load(open(path or f"bench_results/{label}.json"))

    for scheme, scheme_name in SCHEMES:
        print(f"\n## W8 {scheme_name} - A8W8 speedup over Abf16W8, full pipeline\n")
        print("| shape (n x k) | " + " | ".join(f"m={m}" for m in MS) + " | canary |")
        print("|---|" + "--:|" * (len(MS) + 1))
        for shape, (n, k) in SHAPE_NK.items():
            group = f"Metal_A8W_w8_{shape}"
            if f"{group}/a8_full_{scheme}/m16" not in rows:
                continue
            canary = [v for key, v in rows.items() if key.startswith(group) and "canary" in key]
            spread = (max(canary) - min(canary)) / min(canary) * 100 if len(canary) >= 2 else float("nan")
            cells = []
            for m in MS:
                a8 = rows.get(f"{group}/a8_full_{scheme}/m{m}")
                bf16 = rows.get(f"{group}/bf16_full_{scheme}/m{m}")
                cells.append("-" if a8 is None or bf16 is None else f"{bf16 / a8:.2f}")
            print(f"| {n} x {k} | " + " | ".join(cells) + f" | {spread:.2f}% |")

    print("\n## Absolute times, microseconds, full pipeline\n")
    for shape, (n, k) in SHAPE_NK.items():
        group = f"Metal_A8W_w8_{shape}"
        if f"{group}/a8_full_sym/m16" not in rows:
            continue
        print(f"\n**n={n} k={k}**\n")
        print("| pipeline | " + " | ".join(f"{m}x{n}x{k}" for m in MS) + " |")
        print("|---|" + "--:|" * len(MS))
        for key, name in [
            ("bf16w16_full", "bf16 act x bf16 weight"),
            ("bf16_full_sym", "bf16 act x W8 symmetric"),
            ("bf16_full_bias", "bf16 act x W8 affine"),
            ("bf16_full_zp", "bf16 act x W8 asymmetric"),
            ("a8_full_sym", "int8 act (dynamic) x W8 symmetric"),
            ("a8_full_bias", "int8 act (dynamic) x W8 affine"),
            ("a8_full_zp", "int8 act (dynamic) x W8 asymmetric"),
        ]:
            values = [rows.get(f"{group}/{key}/m{m}") for m in MS]
            if all(v is None for v in values):
                continue
            print(f"| {name} | " + " | ".join("-" if v is None else f"{v:.1f}" for v in values) + " |")


if __name__ == "__main__":
    main()
