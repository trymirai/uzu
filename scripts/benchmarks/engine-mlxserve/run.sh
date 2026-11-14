#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
    echo "Usage: bench-mlxserve model.json" >&2
    exit 2
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BENCHMARK_MLXSERVE_BINARY:-}" ]]; then
    PYTHON=("$BENCHMARK_MODEL_PYTHON")
else
    "$PROJECT_DIR/bootstrap.sh"
    BENCHMARK_MLXSERVE_BINARY="$PROJECT_DIR/zig-out/bin/engine-mlxserve"
    PYTHON=(uv run --project "$PROJECT_DIR/.." python)
fi

exec "${PYTHON[@]}" -c '
import os, sys
from common import get_model_arguments
os.execv(sys.argv[1], [sys.argv[1], *get_model_arguments(sys.argv[2])])
' "$BENCHMARK_MLXSERVE_BINARY" "$1"
