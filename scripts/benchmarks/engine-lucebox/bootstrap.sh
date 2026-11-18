#!/usr/bin/env bash
# Builds luce_server at the pinned commit for the local NVIDIA GPU and prints the binary's path.
set -euo pipefail

COMMIT="23926555c1600f4a5dd1f36c3eed3cc10b62885f"
# On WSL's own disk: a CUDA build over the Windows drive's mount is slow.
SOURCE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/uzu-benchmarks/lucebox"
BUILD_DIR="$SOURCE_DIR/server/build"
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d .)"

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    git clone --quiet https://github.com/Luce-Org/lucebox.git "$SOURCE_DIR" >&2
fi
if [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" != "$COMMIT" ]]; then
    git -C "$SOURCE_DIR" fetch --quiet origin "$COMMIT" >&2
    git -C "$SOURCE_DIR" checkout --quiet --detach "$COMMIT" >&2
fi
git -C "$SOURCE_DIR" submodule update --init --recursive --quiet >&2

cmake -S "$SOURCE_DIR/server" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH" >&2
cmake --build "$BUILD_DIR" --target luce_server --parallel "$(nproc)" >&2
echo "$BUILD_DIR/luce_server"
