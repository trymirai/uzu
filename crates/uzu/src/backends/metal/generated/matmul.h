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
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t leading_dimension_a;
  uint32_t leading_dimension_b;
  uint32_t leading_dimension_d;
  uint32_t threadgroups_per_column;
  uint32_t threadgroups_per_row;
  uint32_t swizzle_log;
  uint32_t aligned_inner_iterations;
} GemmParams;

#ifdef __METAL_VERSION__
} // namespace matmul
} // namespace uzu
#endif

#endif // UZU_MATMUL_H
