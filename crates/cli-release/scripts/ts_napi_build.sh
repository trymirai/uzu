#!/usr/bin/env bash
set -euo pipefail

TS_NAPI_PATH="$1"
MANIFEST_PATH="$2"
OUTPUT_PATH="$3"

cd $TS_NAPI_PATH
pnpm i --no-frozen-lockfile
pnpm exec napi build --release --features napi --manifest-path $MANIFEST_PATH --output-dir $OUTPUT_PATH