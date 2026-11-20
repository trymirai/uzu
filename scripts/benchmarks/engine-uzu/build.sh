#!/usr/bin/env bash
set -euo pipefail

cargo build --release --manifest-path "$(dirname -- "${BASH_SOURCE[0]}")/Cargo.toml" >&2
