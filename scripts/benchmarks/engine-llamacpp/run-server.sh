#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build/release-server"
BIN_DIR="$BUILD_DIR/bin"
PARALLEL_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1\n')"

cmake \
    --preset release \
    -S "$PROJECT_DIR" \
    -B "$BUILD_DIR" \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN_DIR" \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON >&2

cmake --build "$BUILD_DIR" \
    --config Release \
    --target llama-server \
    --parallel "$PARALLEL_JOBS" >&2

exec "$BIN_DIR/llama-server" "$@"
