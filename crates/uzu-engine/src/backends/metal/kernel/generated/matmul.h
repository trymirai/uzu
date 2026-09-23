// Auto-generated from gpu_types/matmul - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::matmul {
static constant constexpr uint32_t QUANT_PARAMS_GROUP_OUTPUT_ALIGNMENT = 4;

typedef struct {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t leading_dimension_a;
  uint32_t leading_dimension_b;
  uint32_t scale_output_stride;
  uint32_t scale_group_stride;
  uint32_t zero_point_output_stride;
  uint32_t zero_point_group_stride;
  uint32_t leading_dimension_d;
  uint32_t threadgroups_per_column;
  uint32_t threadgroups_per_row;
  uint32_t aligned_inner_iterations;
  bool use_morton;
  float ab_scale;
} GemmParams;
} // namespace uzu::matmul
