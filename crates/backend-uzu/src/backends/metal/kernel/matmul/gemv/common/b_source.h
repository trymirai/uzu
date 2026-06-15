#pragma once

#include "../../../generated/gemm.h"
#include "full_precision_b_source.h"
#include "quantized_b_source.h"

namespace uzu {
namespace gemm {

template <
    typename BT,
    typename AT,
    typename U,
    GemmBPrologueKind B_PROLOGUE,
    uint GROUP_SIZE,
    uint BITS,
    uint K_SPLIT,
    uint RESULTS_PER_SIMDGROUP,
    bool INPUT_ALIGNED>
struct BSource {
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
      uint simd_lane,
      uint k_slice
  ) {
    if constexpr (B_PROLOGUE == GemmBPrologueKind::FullPrecision) {
      FullPrecisionBSource<BT, AT, U, RESULTS_PER_SIMDGROUP, K_SPLIT, INPUT_ALIGNED>::accumulate(
          result,
          b,
          a,
          in_vec_size,
          out_row,
          batch_idx,
          simd_lane,
          k_slice
      );
    } else {
      QuantizedBSource<BT, AT, U, B_PROLOGUE, GROUP_SIZE, BITS, RESULTS_PER_SIMDGROUP, INPUT_ALIGNED>::accumulate(
          result,
          b,
          scales,
          zero_points,
          biases,
          a,
          in_vec_size,
          out_row,
          batch_idx,
          simd_lane
      );
    }
  }
};

} // namespace gemm
} // namespace uzu
