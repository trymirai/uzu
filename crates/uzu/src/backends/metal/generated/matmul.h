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

typedef struct {
  int32_t M;
  int32_t N;
  int32_t K;
  int32_t leading_dimension_a;
  int32_t leading_dimension_b;
  int32_t leading_dimension_d;
  int32_t threadgroups_per_row;
  int32_t threadgroups_per_column;
  int32_t swizzle_log;
  int32_t aligned_inner_iterations;
} GemmParams;

#ifdef __METAL_VERSION__
} // namespace matmul
} // namespace uzu
#endif

#endif // UZU_MATMUL_H
