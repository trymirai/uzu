#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$PROJECT_DIR/deps/tensorfold"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "TensorFold benchmarks require macOS on Apple Silicon" >&2
    exit 1
fi

source "$PROJECT_DIR/../sources.sh"
benchmark_source tensorfold "$SOURCE_DIR"

export ZIG_GLOBAL_CACHE_DIR="$PROJECT_DIR/.zig-cache/global"
(cd "$SOURCE_DIR" && zig build native -Doptimize=ReleaseFast) >&2
