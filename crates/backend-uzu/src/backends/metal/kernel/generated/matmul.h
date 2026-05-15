// Auto-generated from gpu_types/matmul - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::matmul {
typedef struct {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t leading_dimension_activations;
  uint32_t leading_dimension_weights;
  uint32_t leading_dimension_result;
  uint32_t threadgroups_per_column;
  uint32_t threadgroups_per_row;
  uint32_t aligned_inner_iterations;
  bool use_morton;
  float ab_scale;
} GemmParams;
} // namespace uzu::matmul
