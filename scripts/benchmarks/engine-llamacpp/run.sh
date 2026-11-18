#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PRESET="release"
if [[ "$(uname -s)" == Linux ]]; then
    PRESET="release-cuda"
fi
BUILD_DIR="$PROJECT_DIR/build/$PRESET"
BIN_DIR="$BUILD_DIR/bin"
PARALLEL_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1\n')"

cmake \
    --preset "$PRESET" \
    -S "$PROJECT_DIR" >&2

cmake --build "$BUILD_DIR" \
    --config Release \
    --parallel "$PARALLEL_JOBS" >&2

exec "$BIN_DIR/engine-llamacpp" "$@"
