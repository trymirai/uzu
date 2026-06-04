#pragma once

#include "../../../hadamard_transform/hadamard_transform.h"

namespace uzu {
namespace gemm {

template <typename BT, typename DT, typename U, uint RESULTS_PER_SIMDGROUP>
struct Epilogue {
  static METAL_FUNC void store(
      thread U (&result)[RESULTS_PER_SIMDGROUP],
      device DT* d,
      const device BT* output_bias,
      const device int32_t* hadamard_factors,
      threadgroup U* shared_results,
      float ab_scale,
      GemmDTransform output_transform,
      uint out_row,
      uint out_vec_size,
      uint out_block_idx,
      uint simd_group,
      uint simd_lane,
      bool writer
  ) {
    const bool is_scale = output_transform.contains(GemmDTransform::SCALE);
    const bool is_accumulate =
        output_transform.contains(GemmDTransform::ACCUMULATE);
    const bool is_bias = output_transform.contains(GemmDTransform::BIAS);
    const bool use_hadamard = output_transform.contains(GemmDTransform::RHT);

    if (writer && simd_lane == 0) {
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        U value = result[row];
        if (is_scale) {
          value = static_cast<U>(ab_scale) * value;
        }
        const uint global_row = out_row + row;
        if (is_accumulate && global_row < out_vec_size) {
          value += static_cast<U>(d[row]);
        }
        if (is_bias && global_row < out_vec_size) {
          value += static_cast<U>(output_bias[global_row]);
        }
        result[row] = value;
      }
    }

    if (use_hadamard) {
      if (simd_lane == 0) {
        METAL_PRAGMA_UNROLL
        for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          shared_results[simd_group * RESULTS_PER_SIMDGROUP + row] =
              result[row];
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (simd_group == 0) {
        uint global_out_idx = out_block_idx * 32 + simd_lane;
        if (global_out_idx < out_vec_size) {
          d[simd_lane] = simdgroup_output_random_hadamard_transform(
              static_cast<ushort>(simd_lane),
              static_cast<DT>(shared_results[simd_lane]),
              hadamard_factors[global_out_idx]
          );
        }
      }
    } else {
      if (writer && simd_lane == 0) {
        METAL_PRAGMA_UNROLL
        for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          if (out_row + row < out_vec_size) {
            d[row] = static_cast<DT>(result[row]);
          }
        }
      }
    }
  }
};

} // namespace gemm
} // namespace uzu
