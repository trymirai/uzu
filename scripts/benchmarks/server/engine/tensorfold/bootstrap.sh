#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "TensorFold benchmarks require macOS on Apple Silicon" >&2
    exit 1
fi

# uv fetches the pinned Git source and installs it into an isolated environment.
env -u VIRTUAL_ENV uv sync --project "$PROJECT_DIR" --frozen >&2

# Complete downloads before the server's readiness timeout starts. Local model
# directories need no download; extra arguments can prepare optional drafters.
for model in "$@"; do
    if [[ ! -d "$model" ]]; then
        "$PROJECT_DIR/.venv/bin/tensorfold" pull "$model" >&2
    fi
done
