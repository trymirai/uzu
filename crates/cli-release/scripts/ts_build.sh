#!/usr/bin/env bash
set -euo pipefail

TS_PATH="$1"

cd $TS_PATH
pnpm i --no-frozen-lockfile
pnpm run fix
pnpm run build