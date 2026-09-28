#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$PROJECT_DIR/deps/splash"
BUILD_DIR="$PROJECT_DIR/build/release"
PARALLEL_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1\n')"
PYTHON_EXECUTABLE="${Python3_EXECUTABLE:-$PROJECT_DIR/../.venv/bin/python}"

if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
    PYTHON_EXECUTABLE="$(command -v python3)"
fi

if [[ ! -e "$SOURCE_DIR" ]]; then
    mkdir -p "$PROJECT_DIR/deps"
    git clone --depth 1 --branch 1.1.0 https://github.com/incoai/splash.git "$SOURCE_DIR" >&2
fi

cmake --preset release -S "$PROJECT_DIR" -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE" >&2
cmake --build "$BUILD_DIR" --parallel "$PARALLEL_JOBS" >&2
