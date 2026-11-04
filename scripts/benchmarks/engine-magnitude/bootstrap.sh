#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$PROJECT_DIR/deps/magnitude"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "Magnitude benchmarks require macOS on Apple Silicon" >&2
    exit 1
fi

source "$PROJECT_DIR/../sources.sh"
benchmark_source magnitude "$SOURCE_DIR"

# Build the standalone, in-process engine with the upstream toolchain and lockfile.
cd "$SOURCE_DIR/inference"
CARGO_TARGET_DIR="$PROJECT_DIR/target" cargo build --release --locked \
    -p magnitude-engine-cli --bin magnitude-engine >&2
