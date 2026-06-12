#pragma once

#include "../../common/defines.h"

namespace uzu {
namespace gemm {

template <typename BT, typename AT, typename U, uint RESULTS_PER_SIMDGROUP, uint K_SPLIT, bool INPUT_ALIGNED>
struct FullPrecisionBSource {
  static METAL_FUNC void accumulate(
      thread U (&result)[RESULTS_PER_SIMDGROUP],
      const device uint32_t* b,
      const device AT* a,
      uint in_vec_size,
      uint out_row,
      uint batch_idx,
      uint simd_lane,
      uint k_slice
  ) {
    constexpr uint values_per_thread = 4;
    constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
    typedef vec<BT, 4> W4;
    typedef vec<AT, 4> I4;

    const uint k_stride = K_SPLIT * block_size;
    const uint k_start = k_slice * block_size;
    const device BT* weights = reinterpret_cast<const device BT*>(b);
    weights += out_row * in_vec_size + simd_lane * values_per_thread + k_start;
    const device AT* input = a + batch_idx * in_vec_size + simd_lane * values_per_thread + k_start;

    uint k = k_start;
    for (; k + block_size <= in_vec_size; k += k_stride) {
      float4 input_values = static_cast<float4>(*reinterpret_cast<const device I4*>(input));
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        const device BT* weight_row = weights + row * in_vec_size;
        result[row] += dot(static_cast<float4>(*reinterpret_cast<const device W4*>(weight_row)), input_values);
      }
      weights += k_stride;
      input += k_stride;
    }

    if constexpr (K_SPLIT == 1 && !INPUT_ALIGNED) {
      const uint thread_offset = simd_lane * values_per_thread;
      const int remaining = (k + thread_offset < in_vec_size) ? min(static_cast<int>(in_vec_size - k - thread_offset),
                                                                    static_cast<int>(values_per_thread))
                                                              : 0;
      if (remaining > 0) {
        METAL_PRAGMA_UNROLL
        for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          const device BT* weight_row = weights + row * in_vec_size;
          for (int index = 0; index < remaining; index++) {
            result[row] += static_cast<U>(weight_row[index]) * static_cast<U>(input[index]);
          }
        }
      }
    }
  }
};

} // namespace gemm
} // namespace uzu
