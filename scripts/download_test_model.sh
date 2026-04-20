#!/usr/bin/env bash

set -euo pipefail

ROOT_PATH="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS_PATH="$ROOT_PATH/tools"

cd "$TOOLS_PATH"
uv sync
uv run downloader download meta-llama/Llama-3.2-1B-Instruct