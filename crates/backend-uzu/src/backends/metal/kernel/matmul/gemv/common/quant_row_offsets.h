#pragma once

#include "../../common/quant_pack.h"
#include "gemv_common.h"

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

  void load(thread U* scale, thread U* offset) const {
    scale[0] = static_cast<U>(scales[0]);
    scale[1] = static_cast<U>(scales[group_stride]);
    scale[2] = static_cast<U>(scales[2 * group_stride]);
    scale[3] = static_cast<U>(scales[3 * group_stride]);

    if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
      offset[0] = static_cast<U>(biases[0]);
      offset[1] = static_cast<U>(biases[group_stride]);
      offset[2] = static_cast<U>(biases[2 * group_stride]);
      offset[3] = static_cast<U>(biases[3 * group_stride]);
    } else if constexpr (
        B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant
    ) {
      uchar4 zp_bytes = uchar4(
          zps[0],
          zps[zp_stride],
          zps[2 * zp_stride],
          zps[3 * zp_stride]
      );
      uchar4 zp_nibbles;
      if constexpr (BITS == 4) {
        const uint8_t shift = high_nibble ? 4u : 0u;
        zp_nibbles = (zp_bytes >> shift) & uchar4(0x0F);
      } else {
        zp_nibbles = zp_bytes;
      }
      offset[0] = -scale[0] * static_cast<U>(zp_nibbles.x);
      offset[1] = -scale[1] * static_cast<U>(zp_nibbles.y);
      offset[2] = -scale[2] * static_cast<U>(zp_nibbles.z);
      offset[3] = -scale[3] * static_cast<U>(zp_nibbles.w);
    } else {
      constexpr U midpoint = U(1u << (BITS - 1));
      offset[0] = -scale[0] * midpoint;
      offset[1] = -scale[1] * midpoint;
      offset[2] = -scale[2] * midpoint;
      offset[3] = -scale[3] * midpoint;
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
