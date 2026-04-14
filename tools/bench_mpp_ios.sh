#!/bin/bash
set -euo pipefail

DEVICE="${1:-00008150-001965E22282401C}"

SHAPES=(
  "M\[128\]K\[2048\]"
  "M\[128\]K\[4096\]"
  "M\[256\]K\[4096\]"
  "M\[512\]K\[8192\]"
)

RESULTS_FILE="mpp_bench_results.txt"
> "$RESULTS_FILE"

for shape in "${SHAPES[@]}"; do
  echo ">>> GEMM_MPP / $shape"
  cargo +nightly dinghy -d "$DEVICE" bench -p uzu --bench kernel \
    -- "GEMM_MPP/BF16/$shape" 2>&1 \
    | grep -E "^(Metal/|  +time:|  +thrpt:)" \
    | tee -a "$RESULTS_FILE"
  echo "" >> "$RESULTS_FILE"
done

echo ""
echo "=== Results ==="
cat "$RESULTS_FILE"
