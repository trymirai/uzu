#!/usr/bin/env bash
set -euo pipefail

REVISION="7c58cf0ec5276287b4d341bdfc96ef0fc66360af"
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

if [[ ! -e "$SOURCE_DIR" ]]; then
    mkdir -p "$PROJECT_DIR/deps"
    git clone https://github.com/google-ai-edge/ml-drift.git "$SOURCE_DIR" >&2
    git -C "$SOURCE_DIR" checkout --detach "$REVISION" >&2
fi
if [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" != "$REVISION" ]]; then
    echo "Unexpected ML Drift revision in $SOURCE_DIR; expected $REVISION" >&2
    exit 1
fi

for patch in "$PROJECT_DIR"/patches/*.patch; do
    if ! git -C "$SOURCE_DIR" apply --reverse --check "$patch" 2>/dev/null; then
        git -C "$SOURCE_DIR" apply --check "$patch"
        git -C "$SOURCE_DIR" apply "$patch"
    fi
done

# Bazel cannot reference files outside the workspace, so the adapter and the
# shared benchmark sources become a package of the upstream checkout.
mkdir -p "$ADAPTER_DIR"
for source in "$PROJECT_DIR"/src/* "$PROJECT_DIR"/../common-cpp/src/{bench.hpp,common.cpp,common.hpp,memory_counters.c,memory_counters.h}; do
    destination="$ADAPTER_DIR/$(basename "$source")"
    if ! cmp -s "$source" "$destination"; then
        cp "$source" "$destination"
    fi
done

# glaze needs C++23, and absl must use the same standard as its users.
# Stripped Rust proc-macros have a misaligned string table that dyld rejects.
cd "$SOURCE_DIR"
bazelisk build -c opt \
    --cxxopt=-std=c++23 \
    --@rules_rust//rust/settings:extra_exec_rustc_flags=-Cstrip=none \
    //benchmarks:engine-mldrift >&2
