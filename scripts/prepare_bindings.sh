#!/usr/bin/env bash

set -euo pipefail

ROOT_PATH="$(cd "$(dirname "$0")/.." && pwd)"

if [ -d "$HOME/.cargo/bin" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi
if [ -d "/opt/homebrew/bin" ]; then
    export PATH="/opt/homebrew/bin:$PATH"
fi

cd "$ROOT_PATH"
cargo run -p cli-release prepare-bindings "$@"
