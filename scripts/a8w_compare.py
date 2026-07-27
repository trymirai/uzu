#!/usr/bin/env python3
"""Speedup table against the unquantized bf16 GEMM, with full scheme and shape names.

Usage: a8w_compare.py <label>[=<path.json>] [<shape> ...]

Quantization schemes, all with bf16 accumulate and bf16 output:

  A16W16              bf16 activations x bf16 weights - no quantization anywhere
  A16W8-sym(g64)      bf16 acts x uint8 weights, symmetric, bf16 scale per 64 k-elements
  A16W8-zp(g64)       as above plus a per-group zero point (asymmetric)
  A8(g32)W8(g64)-sym  int8 acts, dynamic f32 scale per 32 k-elements (RHT block size),
                      against the A16W8-sym weights. This is what uzu ships today.
  A8(g32)W8(g64)-zp   as above against the A16W8-zp weights

Both A8 scale axes vary along k, which is why the dequant runs inside the k loop. SOTA
CUDA A8W8 is A8(per-token) x W8(per-channel): neither varies along k, so it dequantizes
once in a single epilogue.

`+prep` rows include the activation preparation kernel: the fused RHT+quantize for A8,
the in-place Hadamard for A16. Everything else is GEMM only.
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

# (benchmark id, display name)
ROWS = [
    ("bf16w16_gemm", "A16W16"),
    ("bf16_gemm_sym", "A16W8-sym(g64)"),
    ("bf16_gemm_zp", "A16W8-zp(g64)"),
    ("a8_gemm_sym", "A8(g32)W8(g64)-sym"),
    ("a8_gemm_zp", "A8(g32)W8(g64)-zp"),
    ("bf16w16_full", "A16W16 +prep"),
    ("bf16_full_sym", "A16W8-sym(g64) +prep"),
    ("a8_full_sym", "A8(g32)W8(g64)-sym +prep"),
]


def main() -> None:
    label, _, path = sys.argv[1].partition("=")
    rows = json.load(open(path or f"bench_results/{label}.json"))
    shapes = sys.argv[2:] or list(SHAPE_NK)

    for shape in shapes:
        group = f"Metal_A8W_w8_{shape}"
        if f"{group}/a8_gemm_sym/m16" not in rows:
            continue
        n, k = SHAPE_NK[shape]
        canary = [v for key, v in rows.items() if key.startswith(group) and "canary" in key]
        spread = (max(canary) - min(canary)) / min(canary) * 100
        print(f"\n## n={n} k={k}   (canary spread {spread:.2f}%)\n")

        print("**Time, microseconds**\n")
        print("| scheme | " + " | ".join(f"{m}x{n}x{k}" for m in MS) + " |")
        print("|---|" + "--:|" * len(MS))
        table = {}
        for key, name in ROWS:
            values = [rows.get(f"{group}/{key}/m{m}") for m in MS]
            if all(v is None for v in values):
                continue
            table[name] = values
            print(f"| {name} | " + " | ".join("-" if v is None else f"{v:.1f}" for v in values) + " |")

        base = table.get("A16W16")
        if not base:
            continue
        print("\n**Speedup vs A16W16 (unquantized bf16 GEMM), >1.00 = faster**\n")
        print("| scheme | " + " | ".join(f"m={m}" for m in MS) + " |")
        print("|---|" + "--:|" * len(MS))
        for name, values in table.items():
            if name == "A16W16":
                continue
            ref = table["A16W16 +prep"] if name.endswith("+prep") and "A16W16 +prep" in table else base
            cells = [
                "-" if v is None or r is None else f"{r / v:.2f}"
                for v, r in zip(values, ref)
            ]
            print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
