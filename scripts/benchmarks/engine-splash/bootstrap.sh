#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$PROJECT_DIR/deps/splash"
PARALLEL_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1\n')"
PYTHON_EXECUTABLE="${Python3_EXECUTABLE:-$PROJECT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
    PYTHON_EXECUTABLE="$(command -v python3)"
fi

source "$PROJECT_DIR/../sources.sh"
benchmark_source splash "$SOURCE_DIR" "$PROJECT_DIR"/patches/*.patch

make -C "$SOURCE_DIR" --no-print-directory \
    "BUILD_ID_PYTHON=$PYTHON_EXECUTABLE" platform-check >&2
make -C "$SOURCE_DIR" --no-print-directory -j "$PARALLEL_JOBS" \
    "BUILD_ID_PYTHON=$PYTHON_EXECUTABLE" build/splash >&2
