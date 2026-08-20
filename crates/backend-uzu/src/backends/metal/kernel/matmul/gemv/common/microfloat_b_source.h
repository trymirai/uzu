#pragma once

#include "../../common/microfloat.h"

namespace uzu {
namespace gemm {

template <
    typename BT,
    typename AT,
    typename U,
    uint GROUP_SIZE,
    uint K_SPLIT,
    uint RESULTS_PER_SIMDGROUP,
    bool INPUT_ALIGNED>
struct MicrofloatBSource {
  static METAL_FUNC void accumulate(
      thread U (&result)[RESULTS_PER_SIMDGROUP],
      const device uint32_t* b,
      const device uint8_t* scales,
      const device BT* global_scales,
      const device AT* a,
      const device uint* gather_indices,
      bool gathered,
      uint in_vec_size,
      uint out_vec_size,
      uint out_row,
      uint assignment_idx,
      uint source_row,
      uint matrix,
      uint simd_lane,
      uint k_slice
  ) {
    constexpr uint values_per_thread = 4;
    constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
    const uint k_stride = K_SPLIT * block_size;
    const uint k_start = k_slice * block_size;
    const uint thread_offset = simd_lane * values_per_thread;
    const uint code_row_stride = in_vec_size / 2;
    const uint scale_row_stride = in_vec_size / GROUP_SIZE;
    const device uint8_t* codes = reinterpret_cast<const device uint8_t*>(b);
    const U global_scale = static_cast<U>(global_scales[matrix]);

    uint k = k_start;
    for (; k + block_size <= in_vec_size; k += k_stride) {
      const uint column = k + thread_offset;
      const device AT* input = a + size_t(source_row) * size_t(in_vec_size) + size_t(column);
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        if (out_row + row >= out_vec_size) {
          continue;
        }
        const size_t gather_index = size_t(assignment_idx) * size_t(out_vec_size) + size_t(out_row) + size_t(row);
        const uint matrix_row = gathered ? gather_indices[gather_index] : out_row + row;
        const size_t bank_row = size_t(matrix) * size_t(out_vec_size) + size_t(matrix_row);
        const device uint8_t* row_codes = codes + bank_row * size_t(code_row_stride);
        const device uint8_t* row_scales = scales + bank_row * size_t(scale_row_stride);
        const uint exponent = row_scales[column / GROUP_SIZE];
        METAL_PRAGMA_UNROLL
        for (uint index = 0; index < values_per_thread; index++) {
          const uint inner = column + index;
          const uint packed = row_codes[inner / 2];
          const uint code = (inner & 1u) == 0u ? packed & 0x0fu : packed >> 4u;
          result[row] += static_cast<U>(input[index]) * static_cast<U>(decode_mxfp4(code, exponent, global_scale));
        }
      }
    }

    if constexpr (K_SPLIT == 1 && !INPUT_ALIGNED) {
      const int remaining = (k + thread_offset < in_vec_size) ? min(static_cast<int>(in_vec_size - k - thread_offset),
                                                                    static_cast<int>(values_per_thread))
                                                              : 0;
      if (remaining > 0) {
        const uint column = k + thread_offset;
        const device AT* input = a + size_t(source_row) * size_t(in_vec_size) + size_t(column);
        METAL_PRAGMA_UNROLL
        for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          if (out_row + row >= out_vec_size) {
            continue;
          }
          const size_t gather_index = size_t(assignment_idx) * size_t(out_vec_size) + size_t(out_row) + size_t(row);
          const uint matrix_row = gathered ? gather_indices[gather_index] : out_row + row;
          const size_t bank_row = size_t(matrix) * size_t(out_vec_size) + size_t(matrix_row);
          const device uint8_t* row_codes = codes + bank_row * size_t(code_row_stride);
          const device uint8_t* row_scales = scales + bank_row * size_t(scale_row_stride);
          const uint exponent = row_scales[column / GROUP_SIZE];
          for (int index = 0; index < remaining; index++) {
            const uint inner = column + static_cast<uint>(index);
            const uint packed = row_codes[inner / 2];
            const uint code = (inner & 1u) == 0u ? packed & 0x0fu : packed >> 4u;
            result[row] += static_cast<U>(input[index]) * static_cast<U>(decode_mxfp4(code, exponent, global_scale));
          }
        }
      }
    }
  }
};

} // namespace gemm
} // namespace uzu
