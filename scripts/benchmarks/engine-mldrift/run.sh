#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
    echo "Usage: bench-mldrift model.json" >&2
    exit 2
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$PROJECT_DIR/deps/ml-drift"
if [[ -n "${BENCHMARK_MLDRIFT_BINARY:-}" ]]; then
    extract_weights() { "$BENCHMARK_MLDRIFT_EXTRACTOR" "$@"; }
else
    "$PROJECT_DIR/bootstrap.sh"
    BENCHMARK_MLDRIFT_BINARY="$SOURCE_DIR/bazel-bin/benchmarks/engine-mldrift"
    BENCHMARK_MLDRIFT_REVISION="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
    extract_weights() { (cd "$SOURCE_DIR" && bazelisk run -c opt //ml_drift/samples/llm:extract_weights_hf -- "$@"); }
fi
WEIGHTS_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/uzu-benchmarks/mldrift/$BENCHMARK_MLDRIFT_REVISION"

MODEL="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"])' "$1")"
PRECISION="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("precision", "q4_0"))' "$1")"
REVISION="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("revision") or "")' "$1")"
case "$PRECISION" in
    q4_0|f16) ;;
    *) echo "Unsupported precision: $PRECISION" >&2; exit 1 ;;
esac

MODEL_PATH="$MODEL"
if [[ -d "$MODEL_PATH" ]]; then
    if [[ -n "$REVISION" ]]; then
        echo "A local model directory has no revision: $MODEL" >&2
        exit 1
    fi
else
    if ! command -v hf >/dev/null; then
        echo "Install the Hugging Face CLI to download $MODEL: uv tool install huggingface_hub" >&2
        exit 1
    fi
    MODEL_PATH="$(hf download --quiet ${REVISION:+--revision "$REVISION"} "$MODEL")"
fi
MODEL_PATH="$(cd -- "$MODEL_PATH" && pwd -P)"

WEIGHTS_DIR="$WEIGHTS_CACHE/$PRECISION/$(/usr/bin/stat -L -f '%N %z %m' "$MODEL_PATH"/*.safetensors | shasum -a 256 | cut -c1-16)"
if [[ ! -d "$WEIGHTS_DIR" ]]; then
    rm -rf "$WEIGHTS_DIR.partial"
    mkdir -p "$WEIGHTS_DIR.partial"
    QUANTIZATION=()
    if [[ "$PRECISION" == q4_0 ]]; then
        QUANTIZATION=(--quantize --embedding_quant_bits=4 --attention_quant_bits=4 --feedforward_quant_bits=4 --block_size=32)
    fi
    extract_weights \
        --model_path="$MODEL_PATH" \
        --output_dir="$WEIGHTS_DIR.partial" \
        ${QUANTIZATION[@]+"${QUANTIZATION[@]}"} >&2
    mv "$WEIGHTS_DIR.partial" "$WEIGHTS_DIR"
fi

exec "$BENCHMARK_MLDRIFT_BINARY" \
    --config="$1" \
    --checkpoint="$MODEL_PATH" \
    --weights="$WEIGHTS_DIR"
