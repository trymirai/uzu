#!/bin/bash

# Reconfigure every run so cached settings cannot bypass these optimizations.
# Static linking lets LTO optimize the host code across library boundaries.
cmake -S src/llamacpp -B build/llamacpp-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$PWD/build/llamacpp-release" \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_LTO=ON \
  -DGGML_NATIVE=ON \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_METAL_NDEBUG=ON \
  -DGGML_METAL_SHADER_DEBUG=OFF \
  -DGGML_ACCELERATE=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=Apple \
  || exit 1

cmake --build build/llamacpp-release \
  --config Release \
  --target benchmark_llamacpp \
  --parallel "$(sysctl -n hw.logicalcpu)" \
  || exit 1

./build/llamacpp-release/benchmark_llamacpp --model "unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_S" --input ./input.json
