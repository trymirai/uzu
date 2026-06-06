#pragma once

#include "gemv_common.h"

namespace uzu {
namespace gemm {

template <typename U, uint K_SPLIT, uint NUM_SIMDGROUPS>
struct Reduce {
  static METAL_FUNC void run(
      thread U (&result)[RESULTS_PER_SIMDGROUP],
      threadgroup U* shared_results,
      uint simd_group,
      uint simd_lane,
      uint row_group,
      uint k_slice
  ) {
    METAL_PRAGMA_UNROLL
    for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
      result[row] = simd_sum(result[row]);
    }

    if constexpr (K_SPLIT > 1) {
      if (simd_lane == 0) {
        METAL_PRAGMA_UNROLL
        for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          shared_results[simd_group * RESULTS_PER_SIMDGROUP + row] = result[row];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (k_slice == 0 && simd_lane == 0) {
        METAL_PRAGMA_UNROLL
        for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          U acc = 0;
          METAL_PRAGMA_UNROLL
          for (uint s = 0; s < K_SPLIT; s++) {
            acc += shared_results[(row_group * K_SPLIT + s) * RESULTS_PER_SIMDGROUP + row];
          }
          result[row] = acc;
        }
      }
    }
  }
};

} // namespace gemm
} // namespace uzu
