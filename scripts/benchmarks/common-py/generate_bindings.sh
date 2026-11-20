#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
uvx --from ctypeslib2==2.4.0 --with clang==21.1.7 clang2py \
    ../common-cpp/src/memory_counters.h ../common-cpp/src/apple_temp_sensors.h ../common-cpp/src/energy_counters.h \
    -l src/mach/_native.*.so |
    sed -e 's/^import ctypes$/import ctypes; from importlib.util import find_spec/' \
        -e "s@ctypes.CDLL('[^']*')@ctypes.CDLL(find_spec(__package__ + '._native').origin)@" > src/mach/_bindings.py
