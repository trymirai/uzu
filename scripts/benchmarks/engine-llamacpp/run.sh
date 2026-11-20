#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname -- "${BASH_SOURCE[0]}")"/build/release/bin/engine-llamacpp "$@"
