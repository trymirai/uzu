#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build/release"
BIN_DIR="$BUILD_DIR/bin"
PARALLEL_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1\n')"

cmake \
    --preset release \
    -S "$PROJECT_DIR" >&2

cmake --build "$BUILD_DIR" \
    --config Release \
    --parallel "$PARALLEL_JOBS" >&2

exec "$BIN_DIR/engine-llamacpp" "$@"
