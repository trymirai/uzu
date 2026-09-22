#!/bin/bash
# Serve a Mirai S package for agent use: builds the CLI if needed, fills in the runtime sidecar the packages
# do not ship, and starts the OpenAI-compatible server with the thinking trace off.
#
#   scripts/serve-mirai-s.sh [package_dir] [port]
#
# Defaults: ~/models/s/q38zmrr2-s-package-v1 on port 8000.
set -euo pipefail

PACKAGE="${1:-$HOME/models/s/q38zmrr2-s-package-v1}"
PORT="${2:-8000}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE="$(cd "$PACKAGE" 2>/dev/null && pwd || { echo "no package directory at $1" >&2; exit 1; })"
ROOT="$(dirname "$PACKAGE")"
NAME="$(basename "$PACKAGE")"

for file in config.json model.safetensors tokenizer.json; do
  [ -f "$PACKAGE/$file" ] || { echo "$PACKAGE is missing $file" >&2; exit 1; }
done

# The packages carry weights, not the runtime's chat encoding; write it once from the config's model family.
# Pass the name as the third argument for a family this does not know.
if [ ! -f "$PACKAGE/encoding.json" ]; then
  FAMILY="${3:-$(python3 - "$PACKAGE/config.json" <<'PY'
import json, sys
layers = json.load(open(sys.argv[1]))["decoder_config"]["transformer_config"]["layer_configs"]
mixers = {layer["mixer_config"].get("type") for layer in layers}
fused = any("qkvg_projection_config" in layer["mixer_config"] for layer in layers)
if mixers == {"DeltaNetConfig"}:
    print("qwen3.8")
elif fused:
    print("muse-glimmer")
else:
    sys.exit("cannot tell the chat encoding from the config; pass it as the third argument")
PY
)}"
  printf '[{"name": "%s", "type": "hanashi"}]\n' "$FAMILY" > "$PACKAGE/encoding.json"
  echo "wrote $PACKAGE/encoding.json ($FAMILY)"
fi

# A Homebrew rustc on PATH shadows the pinned nightly and fails the workspace's rust-version check.
if ! cargo build --release -p cli --manifest-path "$REPO/Cargo.toml" 2>/dev/null; then
  export PATH="$(rustc +nightly --print sysroot)/bin:$HOME/.cargo/bin:$PATH"
  cargo build --release -p cli --manifest-path "$REPO/Cargo.toml"
fi

echo "serving $NAME on http://127.0.0.1:$PORT/v1 (first load reads the whole package from disk)"
exec env LOCAL_PATH="$ROOT" UZU_SERVER_THINKING=0 \
  "$REPO/target/release/cli" server --model "$NAME" --port "$PORT" --host 127.0.0.1
