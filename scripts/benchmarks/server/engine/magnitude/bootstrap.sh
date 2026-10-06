#!/usr/bin/env bash
set -euo pipefail

REVISION="7f65eb322422a4679c6455b4c442ba969f2c3844"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$PROJECT_DIR/deps/magnitude"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "Magnitude benchmarks require macOS on Apple Silicon" >&2
    exit 1
fi

if [[ ! -e "$SOURCE_DIR" ]]; then
    mkdir -p "$PROJECT_DIR/deps"
    git init "$SOURCE_DIR" >&2
    git -C "$SOURCE_DIR" remote add origin https://github.com/magnitudedev/magnitude.git
    git -C "$SOURCE_DIR" fetch --depth 1 origin "$REVISION" >&2
    git -C "$SOURCE_DIR" checkout --detach FETCH_HEAD >&2
fi
if [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" != "$REVISION" ]]; then
    echo "Unexpected Magnitude revision in $SOURCE_DIR; expected $REVISION" >&2
    exit 1
fi

# Build the standalone, in-process engine with the upstream toolchain and lockfile.
cd "$SOURCE_DIR/inference"
CARGO_TARGET_DIR="$PROJECT_DIR/target" cargo build --release --locked \
    -p magnitude-engine-cli --bin magnitude-engine >&2
