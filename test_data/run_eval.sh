#!/usr/bin/env bash
# Run `uzu classify` against every line of the privacy-filter test dataset,
# capture the CLI output verbatim, and dump everything under
# test_data/results/<idx>.txt. Aggregation into results.md is done by
# build_results_md.py — this script is just the runner.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-/tmp/pf_uzu}"
DATASET="${DATASET:-$DIR/privacy_filter_dataset.txt}"
CLI="${CLI:-$DIR/../target/release/cli}"
OUT_DIR="$DIR/results"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

idx=0
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  printf "%s\n" "$line" > "$OUT_DIR/$(printf '%02d' "$idx").input"
  "$CLI" classify "$MODEL" --message "$line" > "$OUT_DIR/$(printf '%02d' "$idx").out" 2>&1
  idx=$((idx + 1))
done < "$DATASET"

echo "Ran $idx examples -> $OUT_DIR"
