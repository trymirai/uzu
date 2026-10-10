#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$PROJECT_DIR/deps/ml-drift"
"$PROJECT_DIR/bootstrap.sh"

MODEL=""
PRECISION="q4_0"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "$1 requires a model directory or Hugging Face repository" >&2
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        --precision)
            if [[ $# -lt 2 || ! "$2" =~ ^(q4_0|f16)$ ]]; then
                echo "$1 requires q4_0 or f16" >&2
                exit 1
            fi
            PRECISION="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done
if [[ -z "$MODEL" ]]; then
    echo "--model is required" >&2
    exit 1
fi

MODEL_PATH="$MODEL"
if [[ ! -d "$MODEL_PATH" ]]; then
    MODEL_PATH="$(uv run --project "$PROJECT_DIR" python -c \
        'import sys; from common import get_model_path; print(get_model_path(sys.argv[1]))' "$MODEL")"
fi
MODEL_PATH="$(cd -- "$MODEL_PATH" && pwd -P)"

WEIGHTS_DIR="$PROJECT_DIR/workspace/$(git -C "$SOURCE_DIR" rev-parse --short=12 HEAD)/$PRECISION"
WEIGHTS_DIR="$WEIGHTS_DIR/$(printf '%s' "$MODEL_PATH" | shasum -a 256 | cut -c1-16)"
if [[ ! -d "$WEIGHTS_DIR" ]]; then
    rm -rf "$WEIGHTS_DIR.partial"
    mkdir -p "$WEIGHTS_DIR.partial"
    QUANTIZATION=()
    if [[ "$PRECISION" == q4_0 ]]; then
        QUANTIZATION=(--quantize --embedding_quant_bits=4 --attention_quant_bits=4 --feedforward_quant_bits=4 --block_size=32)
    fi
    uv run --project "$PROJECT_DIR" "$SOURCE_DIR/ml_drift/samples/llm/extract_weights_hf.py" \
        --model_path "$MODEL_PATH" \
        --output_dir "$WEIGHTS_DIR.partial" \
        ${QUANTIZATION[@]+"${QUANTIZATION[@]}"} >&2
    mv "$WEIGHTS_DIR.partial" "$WEIGHTS_DIR"
fi

exec "$SOURCE_DIR/bazel-bin/benchmarks/engine-mldrift" \
    --checkpoint="$MODEL_PATH" \
    --weights="$WEIGHTS_DIR" \
    --prompt_project="$PROJECT_DIR"
