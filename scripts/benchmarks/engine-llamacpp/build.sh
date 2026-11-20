#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARALLEL_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1\n')"

cmake \
    --preset release \
    -S "$PROJECT_DIR" >&2

cmake --build "$PROJECT_DIR/build/release" \
    --config Release \
    --parallel "$PARALLEL_JOBS" >&2
