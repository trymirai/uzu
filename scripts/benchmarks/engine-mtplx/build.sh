#!/usr/bin/env bash
set -euo pipefail

uv sync --no-active --project "$(dirname -- "${BASH_SOURCE[0]}")" >&2
