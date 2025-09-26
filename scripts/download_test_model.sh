#!/usr/bin/env bash

set -euo pipefail

if [[ "$(uname)" != "Darwin" ]]; then
  echo "Non-macOS platform detected (\"$(uname)\"); skipping test-model download."
  exit 0
fi

SCRIPTS_PATH="$(cd "$(dirname "$0")" && pwd)"
TOOLS_PATH="$SCRIPTS_PATH/tools"

cd $TOOLS_PATH
uv sync
uv run main.py download-model meta-llama/Llama-3.2-1B-Instruct