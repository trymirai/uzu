#!/usr/bin/env bash
# download_test_model.sh
# This script ensures the test model (Meta-Llama-3.2-1B-Instruct-float16)
# required by the test-suite is present in the SDK cache directory.
# Usage: bash download_test_model.sh
#
# The cache root mirrors the logic in `uzu/tests/common/mod.rs`:
#   • On macOS/iOS  →  "$HOME/Library/Caches/com.mirai.sdk.storage"
#
set -euo pipefail

# Exit early on non-macOS platform …
if [[ "$(uname)" != "Darwin" ]]; then
  echo "Non-macOS platform detected (\"$(uname)\"); skipping test-model download."
  exit 0
fi

# macOS cache directory root
ROOT_DIR="$HOME/Library/Caches/com.mirai.sdk.storage"

# Create model directory
MODEL_DIR="$ROOT_DIR/Meta-Llama-3.2-1B-Instruct-float16"
mkdir -p "$MODEL_DIR"

echo "Model directory: $MODEL_DIR"

# List of "<filename> <url>" pairs (compatible with macOS Bash 3.2 – no associative arrays)
FILES=(
  "config.json https://artifacts.trymirai.com/models/0.1.0/float16/Meta-Llama-3.2-1B-Instruct/config.json"
  "model.safetensors https://artifacts.trymirai.com/models/0.1.0/float16/Meta-Llama-3.2-1B-Instruct/model.safetensors"
  "tokenizer.json https://artifacts.trymirai.com/models/0.1.0/float16/Meta-Llama-3.2-1B-Instruct/tokenizer.json"
  "tokenizer_config.json https://artifacts.trymirai.com/models/0.1.0/float16/Meta-Llama-3.2-1B-Instruct/tokenizer_config.json"
  "traces.safetensors https://artifacts.trymirai.com/models/0.1.0/float16/Meta-Llama-3.2-1B-Instruct/traces.safetensors"
)

for ITEM in "${FILES[@]}"; do
  NAME="${ITEM%% *}"
  URL="${ITEM#* }"
  DEST="$MODEL_DIR/$NAME"
  PART="$DEST.part"

  if [[ -f "$DEST" ]]; then
    echo "✓ $NAME already present — skipping download"
    continue
  fi

  # If a partial download exists, resume it
  if [[ -f "$PART" ]]; then
    echo "↻ Resuming ${NAME} download…"
  else
    echo "↓ Downloading ${NAME}…"
  fi

  # Use --continue-at - (alias -C -) to resume; always write to .part file first
  curl -L --fail --progress-bar -C - "$URL" -o "$PART"

  # On successful download, rename to final file name
  mv -f "$PART" "$DEST"
done

echo "✔ All required artifacts are available in $MODEL_DIR" 