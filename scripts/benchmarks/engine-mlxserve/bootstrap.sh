#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$PROJECT_DIR/deps/mlx-serve"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "mlx-serve benchmarks require macOS on Apple Silicon" >&2
    exit 1
fi

source "$PROJECT_DIR/../sources.sh"
benchmark_source mlxserve "$SOURCE_DIR" "$PROJECT_DIR"/patches/*.patch

# Only source dependencies of the MLX backend. No released inference binaries.
git -C "$SOURCE_DIR" submodule update --init --depth 1 \
    lib/mlx-src lib/mlxc-src >&2
"$SOURCE_DIR/scripts/build-mlx.sh" >&2

cp "$PROJECT_DIR"/src/* "$SOURCE_DIR/src/"
cp "$PROJECT_DIR/build.zig" "$SOURCE_DIR/build-benchmarks.zig"
cd "$SOURCE_DIR"
export ZIG_GLOBAL_CACHE_DIR="$PROJECT_DIR/.zig-cache/global"
zig build --build-file build-benchmarks.zig --prefix "$PROJECT_DIR/zig-out" -Doptimize=ReleaseFast -j8 >&2
