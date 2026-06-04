#pragma once

#include "../../common/quant_pack.h"

namespace uzu {
namespace gemm {

template <typename BT, typename U, GemmBPrologueKind B_PROLOGUE, uint BITS>
struct QuantRowOffsets {
  static_assert(
      BITS == 4 || BITS == 8,
      "QuantRowOffsets supports 4- and 8-bit only"
  );
  const device BT* scales = nullptr;
  const device BT* biases = nullptr;
  const device uint8_t* zps = nullptr;
  uint group_stride = 0;
  uint zp_stride = 0;
  bool high_nibble = false;

  template <uint RESULTS_PER_SIMDGROUP>
  void load(
      thread U (&scale)[RESULTS_PER_SIMDGROUP],
      thread U (&offset)[RESULTS_PER_SIMDGROUP]
  ) const {
    METAL_PRAGMA_UNROLL
    for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
      scale[row] = static_cast<U>(scales[row * group_stride]);
    }

    if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        offset[row] = static_cast<U>(biases[row * group_stride]);
      }
    } else if constexpr (
        B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant
    ) {
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        uint8_t zp = zps[row * zp_stride];
        if constexpr (BITS == 4) {
          const uint8_t shift = high_nibble ? 4u : 0u;
          zp = (zp >> shift) & 0x0F;
        }
        offset[row] = -scale[row] * static_cast<U>(zp);
      }
    } else {
      constexpr U midpoint = U(1u << (BITS - 1));
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        offset[row] = -scale[row] * midpoint;
      }
    }
  }

  void advance(uint groups) {
    scales += groups;
    if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
      biases += groups;
    } else if constexpr (
        B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant
    ) {
      constexpr uint zps_per_byte = get_pack_factor<BITS, 8>();
      zps += groups / zps_per_byte;
    }
  }
};

} // namespace gemm
} // namespace uzu
