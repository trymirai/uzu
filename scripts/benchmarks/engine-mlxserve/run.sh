#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$PROJECT_DIR/bootstrap.sh"

ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model|-d|--draft-model)
            FLAG="$1"
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "$FLAG requires a model directory or Hugging Face repository" >&2
                exit 1
            fi
            MODEL_PATH="$2"
            if [[ ! -d "$MODEL_PATH" ]]; then
                # Resolve both checkpoints before native inference. Python only downloads.
                MODEL_PATH="$(uv run --project "$PROJECT_DIR/.." python -c \
                    'import sys; from common import get_model_path; print(get_model_path(sys.argv[1]))' "$MODEL_PATH")"
            fi
            ARGS+=("$FLAG" "$MODEL_PATH")
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
exec "$PROJECT_DIR/zig-out/bin/engine-mlxserve" "${ARGS[@]}"
