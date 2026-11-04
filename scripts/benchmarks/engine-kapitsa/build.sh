#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$PROJECT_DIR/deps/kapitsa"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "Kapitsa benchmarks require macOS on Apple Silicon" >&2
    exit 1
fi

source "$PROJECT_DIR/../sources.sh"
benchmark_source kapitsa "$SOURCE_DIR" "$PROJECT_DIR"/patches/*.patch

cmake -S "$SOURCE_DIR" -B "$SOURCE_DIR/build" -DCMAKE_BUILD_TYPE=Release -DKAPITSA_BUILD_SDK=ON >&2
cmake --build "$SOURCE_DIR/build" --target kapitsa_sdk -j "$(sysctl -n hw.ncpu)" >&2

uv sync --no-active --project "$PROJECT_DIR" >&2
