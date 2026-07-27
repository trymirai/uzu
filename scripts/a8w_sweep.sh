#!/usr/bin/env bash
# Drives an a8w bench sweep one criterion group per process, snapshotting results
# between chunks. Process isolation also resets allocator/PSO-cache state.
#
#   ./scripts/a8w_sweep.sh <label> <filter> [<filter> ...]
#
# Results land in bench_results/<label>.json (one entry per benchmark id).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="$1"; shift
OUT_DIR="$REPO/bench_results"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/$LABEL.json"

export UZU_A8W_COOLOFF_MS="${UZU_A8W_COOLOFF_MS:-3000}"
GAP="${A8W_CHUNK_GAP_S:-15}"

# Compile first, measure second - never concurrently.
cargo bench -p backend-uzu --lib --no-run >/dev/null 2>&1

# Anything already in target/criterion predates this sweep; without this cutoff a chunk
# that has not run yet would contribute a previous sweep's numbers as if they were fresh.
STARTED_AT="$(python3 -c 'import time; print(time.time())')"

echo "{}" > "$OUT"
first=1
for filter in "$@"; do
  if [ $first -eq 0 ]; then
    echo "== chunk gap ${GAP}s =="
    sleep "$GAP"
  fi
  first=0
  echo "== chunk: $filter =="
  cargo bench -p backend-uzu --lib "$filter" 2>&1 | grep -E "^Metal/|time:" || true
  python3 "$REPO/scripts/a8w_collect.py" "$REPO/target/criterion" "$OUT" "$STARTED_AT"
done

echo "== wrote $OUT =="
python3 - "$OUT" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
for key in sorted(rows):
    print(f"{rows[key]:10.2f}  {key}")
PY
