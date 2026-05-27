// Auto-generated from gpu_types/matmul - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::matmul {
typedef struct {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t leading_dimension_a;
  uint32_t leading_dimension_b;
  uint32_t leading_dimension_d;
  uint32_t threadgroups_per_column;
  uint32_t threadgroups_per_row;
  uint32_t aligned_inner_iterations;
  bool use_morton;
  float ab_scale;
} GemmParams;

typedef struct {
  uint32_t in_vec_size;
  uint32_t out_vec_size;
  uint32_t batch_size;
  uint32_t matrix_leading_dimension;
  uint32_t output_rows_per_threadgroup;
  float ab_scale;
} GemvParams;
} // namespace uzu::matmul
