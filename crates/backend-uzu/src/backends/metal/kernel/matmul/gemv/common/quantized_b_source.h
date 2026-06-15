#pragma once

#include "../../common/qdot.h"
#include "../../common/quant_pack.h"
#include "quantized_row_state.h"

namespace uzu {
namespace gemm {

template <
    typename BT,
    typename AT,
    typename U,
    GemmBPrologueKind B_PROLOGUE,
    uint GROUP_SIZE,
    uint BITS,
    uint RESULTS_PER_SIMDGROUP,
    bool INPUT_ALIGNED>
struct QuantizedBSource {
  static METAL_FUNC void accumulate(
      thread U (&result)[RESULTS_PER_SIMDGROUP],
      const device uint32_t* b,
      const device BT* scales,
      const device uint8_t* zero_points,
      const device BT* biases,
      const device AT* a,
      uint in_vec_size,
      uint out_row,
      uint batch_idx,
      uint simd_lane
  ) {
    constexpr uint pack_factor = get_pack_factor<BITS, 32>();
    constexpr uint bytes_per_pack = get_bytes_per_pack<BITS, 32>();
    constexpr uint packs_per_thread = 2;
    constexpr uint values_per_thread = pack_factor * packs_per_thread;
    constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
    constexpr uint scale_step_per_thread = GROUP_SIZE / values_per_thread;
    using RowState = QuantizedRowState<BT, U, B_PROLOGUE, BITS, RESULTS_PER_SIMDGROUP>;
    using RowParams = typename RowState::Params;

    const uint weights_row_stride = in_vec_size * bytes_per_pack / pack_factor;
    const uint group_count = (in_vec_size + GROUP_SIZE - 1) / GROUP_SIZE;
    const uint group_offset = simd_lane / scale_step_per_thread;
    const device uint8_t* weights = reinterpret_cast<const device uint8_t*>(b);
    weights += out_row * weights_row_stride + simd_lane * packs_per_thread * bytes_per_pack;

    RowState row_state(scales, zero_points, biases, out_row, group_count, group_offset);

    const device AT* input = a + batch_idx * in_vec_size + simd_lane * values_per_thread;
    thread U input_values[values_per_thread];

    uint k = 0;
    for (; k + block_size <= in_vec_size; k += block_size) {
      U input_sum = load_vector<AT, U, values_per_thread, BITS>(input, input_values);

      RowParams row_params;
      row_state.load(row_params);
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        const device uint8_t* weight_row = weights + row * weights_row_stride;
        result[row] += qdot<U, values_per_thread, BITS>(
            weight_row,
            input_values,
            row_params.scale[row],
            row_params.offset[row],
            input_sum
        );
      }

      weights += block_size * bytes_per_pack / pack_factor;
      row_state.advance(block_size / GROUP_SIZE);
      input += block_size;
    }

    if constexpr (!INPUT_ALIGNED) {
      const uint thread_offset = simd_lane * values_per_thread;
      const int remaining = (k + thread_offset < in_vec_size) ? min(static_cast<int>(in_vec_size - k - thread_offset),
                                                                    static_cast<int>(values_per_thread))
                                                              : 0;
      if (remaining > 0) {
        U input_sum = load_vector_safe<AT, U, values_per_thread>(input, input_values, remaining);

        RowParams row_params;
        row_state.load(row_params);
        METAL_PRAGMA_UNROLL
        for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          const device uint8_t* weight_row = weights + row * weights_row_stride;
          result[row] += qdot_safe<U, values_per_thread, BITS>(
              weight_row,
              input_values,
              row_params.scale[row],
              row_params.offset[row],
              input_sum,
              remaining
          );
        }
      }
    }
  }
};

} // namespace gemm
} // namespace uzu
