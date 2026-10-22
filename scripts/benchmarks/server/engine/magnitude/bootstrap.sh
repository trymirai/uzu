#!/usr/bin/env bash
set -euo pipefail

VERSION="0.2.6"
REVISION="38f0adb8f6470642aaa15f7aa46ac2d135359d23"
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
fi
if [[ "$(git -C "$SOURCE_DIR" rev-parse --verify HEAD 2>/dev/null || true)" != "$REVISION" ]]; then
    if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
        echo "Cannot switch Magnitude to $VERSION: local changes in $SOURCE_DIR" >&2
        exit 1
    fi
    git -C "$SOURCE_DIR" fetch --depth 1 origin "refs/tags/@magnitudedev/cli@$VERSION" >&2
    if [[ "$(git -C "$SOURCE_DIR" rev-parse 'FETCH_HEAD^{commit}')" != "$REVISION" ]]; then
        echo "Unexpected Magnitude $VERSION revision; expected $REVISION" >&2
        exit 1
    fi
    git -C "$SOURCE_DIR" checkout --detach "$REVISION" >&2
fi

# Build the standalone, in-process engine with the upstream toolchain and lockfile.
cd "$SOURCE_DIR/inference"
CARGO_TARGET_DIR="$PROJECT_DIR/target" cargo build --release --locked \
    -p magnitude-engine-cli --bin magnitude-engine >&2
