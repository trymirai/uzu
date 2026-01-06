// Shared types for GEMM kernel - used by both Metal shaders and Rust (via
// bindgen). This header must be C-compatible for bindgen to parse it.

#pragma once

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
// GEMM Parameters
///////////////////////////////////////////////////////////////////////////////

/// Parameters for the main GEMM kernel.
/// Layout: 8 x int32, 3 x int64, 2 x int32 = 64 bytes total
struct GEMMParams {
  /// M dimension - batch/number of tokens (rows of A, rows of D)
  int batch;
  /// N dimension - output_dim (cols of B, cols of D)
  int output_dim;
  /// K dimension - input_dim/reduction dimension (cols of A, rows of B)
  int input_dim;

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
};

/// Parameters for addmm (alpha * A @ B + beta * C) operations
struct GEMMAddMMParams {
  int ldc;
  int fdc;

  int64_t batch_stride_c;

  float alpha;
  float beta;
};

struct SplitKGEMMParams {
  int m;
  int n;
  int k;

  int leading_dim_a;
  int leading_dim_b;
  int leading_dim_accumulator;

  int tile_count_n;
  int tile_count_m;

  int partition_count;
  int output_elements_per_partition;
  int k_elements_per_partition;
  int gemm_k_iterations_aligned;
};

// Close Metal namespace
#ifdef __METAL_VERSION__
} // namespace matmul
} // namespace uzu
#endif
