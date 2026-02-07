// Auto-generated from gpu_types/matmul.rs - do not edit manually
#pragma once

#ifndef UZU_MATMUL_H
#define UZU_MATMUL_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

namespace uzu {
namespace matmul {
#else
#include <stdint.h>
#endif

/**Main GEMM parameters passed to Metal kernels as constant buffer. */
typedef struct {
  int32_t M;
  int32_t N;
  int32_t K;
  int32_t lda;
  int32_t ldb;
  int32_t ldd;
  int32_t tiles_n;
  int32_t tiles_m;
  int64_t batch_stride_a;
  int64_t batch_stride_b;
  int64_t batch_stride_d;
  int32_t swizzle_log;
  int32_t gemm_k_iterations_aligned;
  int32_t batch_ndim;
} GEMMParams;

/**Split-K GEMM parameters. */
typedef struct {
  int32_t M;
  int32_t N;
  int32_t K;
  int32_t lda;
  int32_t ldb;
  int32_t ldc;
  int32_t tiles_n;
  int32_t tiles_m;
  int32_t split_k_partitions;
  int32_t split_k_partition_stride;
  int32_t split_k_partition_size;
  int32_t gemm_k_iterations_aligned;
} GEMMSpiltKParams;

/**Split-K MLP Fused GEMM parameters. */
typedef struct {
  int32_t M;
  int32_t N;
  int32_t K;
  int32_t lda;
  int32_t ldb;
  int32_t ldc;
  int32_t tiles_n;
  int32_t tiles_m;
  int32_t split_k_partitions;
  int32_t split_k_partition_stride;
  int32_t split_k_partition_size;
  int32_t gemm_k_iterations_aligned;
  int32_t hidden_dim;
} GEMMSpiltKMlpFusedParams;

/**AddMM parameters (alpha * A @ B + beta * C). */
typedef struct {
  int32_t ldc;
  int32_t fdc;
  int64_t batch_stride_c;
  float alpha;
  float beta;
} GEMMAddMMParams;

#ifdef __METAL_VERSION__
} // namespace matmul
} // namespace uzu
#endif

#endif // UZU_MATMUL_H
