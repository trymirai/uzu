#!/usr/bin/env bash
set -euo pipefail

VERSION="v1.2.2"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$PROJECT_DIR/deps/tensorfold"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "TensorFold benchmarks require macOS on Apple Silicon" >&2
    exit 1
fi

if [[ ! -e "$SOURCE_DIR" ]]; then
    mkdir -p "$PROJECT_DIR/deps"
    git clone --depth 1 --branch "$VERSION" https://github.com/ashhart/TensorFold.git "$SOURCE_DIR" >&2
fi
if [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" != "$(git -C "$SOURCE_DIR" rev-parse "refs/tags/$VERSION^{commit}")" ]]; then
    echo "Unexpected TensorFold revision in $SOURCE_DIR; expected $VERSION" >&2
    exit 1
fi

export ZIG_GLOBAL_CACHE_DIR="$PROJECT_DIR/.zig-cache/global"
(cd "$SOURCE_DIR" && zig build native -Doptimize=ReleaseFast) >&2

# Complete downloads before the server's readiness timeout starts. Local model
# directories need no download; extra arguments can prepare optional drafters.
for model in "$@"; do
    if [[ ! -d "$model" ]]; then
        "$SOURCE_DIR/zig-out/native/bin/tensorfold-native" pull "$model" >&2
    fi
done
