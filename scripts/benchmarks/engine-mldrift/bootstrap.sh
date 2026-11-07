#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$PROJECT_DIR/deps/ml-drift"
ADAPTER_DIR="$SOURCE_DIR/benchmarks"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "ML Drift benchmarks require macOS on Apple Silicon" >&2
    exit 1
fi
if ! command -v bazelisk >/dev/null; then
    echo "ML Drift builds with Bazel; install bazelisk (brew install bazelisk)" >&2
    exit 1
fi

source "$PROJECT_DIR/../bootstrap.sh"
benchmark_source mldrift "$SOURCE_DIR"
benchmark_patches "$SOURCE_DIR" "$PROJECT_DIR/patches"
benchmark_stage "$ADAPTER_DIR" "$PROJECT_DIR"/src/* "$PROJECT_DIR"/../common-cpp/src/*

cd "$SOURCE_DIR"
bazelisk build -c opt //benchmarks:engine-mldrift >&2
