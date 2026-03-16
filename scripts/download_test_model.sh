#!/usr/bin/env bash

set -euo pipefail

ROOT_PATH="$(cd "$(dirname "$0")/.." && pwd)"
HELPERS_PATH="$ROOT_PATH/tools/helpers"

cd "$HELPERS_PATH"
uv sync
uv run python main.py download-model meta-llama/Llama-3.2-1B-Instruct