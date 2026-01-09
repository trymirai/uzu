// Shared types for GEMM kernel - used by both Metal shaders and Rust (via
// bindgen). This header must be C-compatible for bindgen to parse it.
// Keep an include guard to avoid clang's "pragma once in main file" warning.
#ifndef UZU_MATMUL_SHARED_TYPES_H
#define UZU_MATMUL_SHARED_TYPES_H

// Metal and C have different type systems
// __METAL_VERSION__ is defined when compiling with the Metal shader compiler
#ifdef __METAL_VERSION__
// Metal shading language - int64_t comes from metal_stdlib
#include <metal_stdlib>
using namespace metal;

namespace uzu {
namespace matmul {

#else
// C/C++ for bindgen - use stdint types
#include <stdint.h>

#endif

///////////////////////////////////////////////////////////////////////////////
// GEMM Parameters (mirrors steel/gemm/params.h, but C-compatible)
///////////////////////////////////////////////////////////////////////////////

// Main GEMM parameters
struct GEMMParams {
  int M;
  int N;
  int K;

  int lda;
  int ldb;
  int ldd;

  int tiles_n;
  int tiles_m;

  int64_t batch_stride_a;
  int64_t batch_stride_b;
  int64_t batch_stride_d;

  int swizzle_log;
  int gemm_k_iterations_aligned;

  int batch_ndim;
};

// Split-K GEMM parameters
struct GEMMSpiltKParams {
  int M;
  int N;
  int K;

  int lda;
  int ldb;
  int ldc;

  int tiles_n;
  int tiles_m;

  int split_k_partitions;
  int split_k_partition_stride;
  int split_k_partition_size;

  int gemm_k_iterations_aligned;
};

// AddMM parameters (alpha * A @ B + beta * C)
struct GEMMAddMMParams {
  int ldc;
  int fdc;

  int64_t batch_stride_c;

  float alpha;
  float beta;
};

// Close Metal namespace
#ifdef __METAL_VERSION__
} // namespace matmul
} // namespace uzu
#endif

#endif // UZU_MATMUL_SHARED_TYPES_H
