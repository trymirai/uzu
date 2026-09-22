#!/bin/bash

if [[ ! -d "./build/llamacpp-release" ]]; then
  cmake -S src/llamacpp -B build/llamacpp-release \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
    -DGGML_NATIVE=ON \
    -DGGML_METAL=ON \
    -DGGML_METAL_NDEBUG=ON \
    -DGGML_ACCELERATE=ON \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=Apple \
    || exit1
fi

cmake --build build/llamacpp-release \
  --config Release \
  --target benchmark_llamacpp \
  --parallel "$(sysctl -n hw.logicalcpu)" \
  || exit1

./build/llamacpp-release/benchmark_llamacpp