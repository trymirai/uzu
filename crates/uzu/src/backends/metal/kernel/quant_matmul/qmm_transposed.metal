#include <metal_stdlib>
#include "../common/dsl.h"
#include "../hadamard_transform/hadamard_transform.h"
#include "quant_matmul.h"

template <typename T, uint GROUP_SIZE, uint BITS>
VARIANTS(T, float, half, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4, 8)
PUBLIC KERNEL(QuantizedMatmulQmmTransposed)(
    const device uint32_t* weights,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device T* input,
    device T* output,
    const device int32_t* hadamard_factors OPTIONAL(use_hadamard),
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    threadgroup T Xs[32 * (32 + 16 / sizeof(T))],
    threadgroup T Ws[32 * (32 + 16 / sizeof(T))],
    const bool use_zero_points SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    const bool aligned_n SPECIALIZE,
    const uint out_block_idx GROUPS(out_vec_size.div_ceil(32)),
    const uint batch_block_idx GROUPS(batch_size.div_ceil(32)),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(4)
) {
  if (use_mlx_quant) {
    if (aligned_n) {
      qmm_transposed_impl<T, GROUP_SIZE, BITS, true, 32, 32, 32, true>(
          weights,
          scales,
          zero_points,
          biases,
          input,
          output,
          Xs,
          Ws,
          in_vec_size,
          out_vec_size,
          batch_size,
          out_block_idx,
          batch_block_idx,
          simd_group,
          simd_lane
      );
    } else {
      qmm_transposed_impl<T, GROUP_SIZE, BITS, false, 32, 32, 32, true>(
          weights,
          scales,
          zero_points,
          biases,
          input,
          output,
          Xs,
          Ws,
          in_vec_size,
          out_vec_size,
          batch_size,
          out_block_idx,
          batch_block_idx,
          simd_group,
          simd_lane
      );
    }
  } else {
    if (aligned_n) {
      qmm_transposed_impl<T, GROUP_SIZE, BITS, true, 32, 32, 32, false>(
          weights,
          scales,
          zero_points,
          biases,
          input,
          output,
          Xs,
          Ws,
          in_vec_size,
          out_vec_size,
          batch_size,
          out_block_idx,
          batch_block_idx,
          simd_group,
          simd_lane
      );
    } else {
      qmm_transposed_impl<T, GROUP_SIZE, BITS, false, 32, 32, 32, false>(
          weights,
          scales,
          zero_points,
          biases,
          input,
          output,
          Xs,
          Ws,
          in_vec_size,
          out_vec_size,
          batch_size,
          out_block_idx,
          batch_block_idx,
          simd_group,
          simd_lane
      );
    }
  }

  if (use_hadamard) {
    threadgroup_barrier(mem_flags::mem_device);

    constexpr uint BLOCK_SIZE = 32;
    constexpr uint NUM_SIMD_GROUPS = 4;

    constexpr uint ROWS_PER_SIMD_GROUP = BLOCK_SIZE / NUM_SIMD_GROUPS;

    const uint out_col = out_block_idx * BLOCK_SIZE + simd_lane;
    const int32_t factor = hadamard_factors[out_col];

    const uint row_start =
        batch_block_idx * BLOCK_SIZE + simd_group * ROWS_PER_SIMD_GROUP;
    const uint row_end = min(row_start + ROWS_PER_SIMD_GROUP, batch_size);

    for (uint row = row_start; row < row_end; row++) {
      uint output_index = row * out_vec_size + out_col;

      output[output_index] = simdgroup_random_hadamard_transform(
          static_cast<ushort>(simd_lane),
          output[output_index],
          factor
      );
    }
  }
}
