#!/usr/bin/env bash
# download_test_model.sh
#
set -euo pipefail

# Exit early on non-macOS platform …
if [[ "$(uname)" != "Darwin" ]]; then
  echo "Non-macOS platform detected (\"$(uname)\"); skipping test-model download."
  exit 0
fi

# Optional first argument specifies the root directory where the model
# should be stored. Defaults to <project-root>/../mirai_models
#   Usage: bash download_test_model.sh [ROOT_DIR]

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT_DIR="${1:-$PROJECT_ROOT/../mirai_models}"

MODEL_DIR="$ROOT_DIR/Meta-Llama-3.2-1B-Instruct-float16"

mkdir -p "$MODEL_DIR"

echo "Model directory: $MODEL_DIR"

HF_BASE="https://huggingface.co/trymirai/Llama-3.2-1B-Instruct-float16/resolve/main"
FILES=(
  "config.json ${HF_BASE}/config.json"
  "model.safetensors ${HF_BASE}/model.safetensors"
  "tokenizer.json ${HF_BASE}/tokenizer.json"
  "tokenizer_config.json ${HF_BASE}/tokenizer_config.json"
  "traces.safetensors ${HF_BASE}/traces.safetensors"
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