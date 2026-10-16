#!/usr/bin/env bash
set -euo pipefail

VERSION="v26.10.1"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$PROJECT_DIR/deps/mlx-serve"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "mlx-serve benchmarks require macOS on Apple Silicon" >&2
    exit 1
fi

if [[ ! -e "$SOURCE_DIR" ]]; then
    mkdir -p "$PROJECT_DIR/deps"
    git clone --depth 1 --branch "$VERSION" https://github.com/ddalcu/mlx-serve.git "$SOURCE_DIR" >&2
fi
if [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" != "$(git -C "$SOURCE_DIR" rev-parse "refs/tags/$VERSION^{commit}")" ]]; then
    echo "Unexpected mlx-serve revision in $SOURCE_DIR; expected $VERSION" >&2
    exit 1
fi

# Only source dependencies of the MLX backend. No released inference binaries.
git -C "$SOURCE_DIR" submodule update --init --depth 1 \
    lib/mlx-src lib/mlxc-src lib/sushi lib/mlx-serve-gguf >&2
"$SOURCE_DIR/scripts/fetch-zig.sh" >&2
"$SOURCE_DIR/scripts/build-mlx.sh" >&2

for patch in "$PROJECT_DIR"/patches/*.patch; do
    if ! git -C "$SOURCE_DIR" apply --reverse --check "$patch" 2>/dev/null; then
        git -C "$SOURCE_DIR" apply --check "$patch"
        git -C "$SOURCE_DIR" apply "$patch"
    fi
done

# Place the adapter in the upstream module so relative Zig imports keep their
# original identity. Do not overwrite or duplicate upstream implementation files.
for source in "$PROJECT_DIR"/src/*.zig; do
    destination="$SOURCE_DIR/src/$(basename "$source")"
    if ! cmp -s "$source" "$destination"; then
        cp "$source" "$destination"
    fi
done

cd "$PROJECT_DIR"
export ZIG_GLOBAL_CACHE_DIR="$PROJECT_DIR/.zig-cache/global"
"$SOURCE_DIR/.zig-toolchain/zig" build -Doptimize=ReleaseFast -j8 >&2
