#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
uvx --from ctypeslib2==2.4.0 --with clang==21.1.7 clang2py \
    deps/splash/runtime/benchmarks.h -l deps/splash/build/splash |
    sed -e 's/^import ctypes$/import ctypes; from install.paths import BINARY as library_path/' \
        -e "s@ctypes.CDLL('[^']*')@ctypes.CDLL(library_path)@" > splash_bindings.py
