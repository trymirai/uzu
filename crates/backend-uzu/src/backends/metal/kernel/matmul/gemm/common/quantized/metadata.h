#pragma once

#include <metal_stdlib>

#include "../../../../generated/gemm.h"
#include "../../../common/quant_unpack.h"

using namespace metal;

namespace uzu {
namespace gemm {

// Per-thread replicated metadata. Compact row storage must consume each row
// sum in the first split-K partition.
template <ushort TILES, ushort SLOTS_PER_TILE, ushort FRAGMENT_EXTENT, ushort SLOT_STRIDE>
struct QuantizationLineCache {
  float values[TILES * SLOTS_PER_TILE];

  METAL_FUNC thread float& slot(const ushort tile, const ushort index) thread {
    return values[tile * SLOTS_PER_TILE + index];
  }

  METAL_FUNC void clear() thread {
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < TILES * SLOTS_PER_TILE; ++index) {
      values[index] = 0.0f;
    }
  }

  METAL_FUNC float at(const short offset) const thread {
    const ushort tile = ushort(offset) / FRAGMENT_EXTENT;
    const ushort index = (ushort(offset) % FRAGMENT_EXTENT) / SLOT_STRIDE;
    return values[tile * SLOTS_PER_TILE + index];
  }

  template <bool ALIGNED, typename Sink>
  static METAL_FUNC void for_each_line(
      const short origin,
      const short limit,
      const uint abs_base,
      const uint groups_per_row,
      const uint group_index,
      Sink sink
  ) {
    METAL_PRAGMA_UNROLL
    for (ushort tile = 0; tile < TILES; ++tile) {
      METAL_PRAGMA_UNROLL
      for (ushort index = 0; index < SLOTS_PER_TILE; ++index) {
        const short coord = origin + tile * FRAGMENT_EXTENT + index * SLOT_STRIDE;
        if (ALIGNED || coord < limit) {
          const uint line_index = abs_base + uint(coord);
          sink(tile, index, line_index * groups_per_row + group_index, line_index);
        }
      }
    }
  }
};

// Symmetric formats omit correction storage entirely.
struct NoCorrectionCache {};

template <typename RightOperand>
struct Correction {
  using Format = typename RightOperand::Format;

  static METAL_FUNC float midpoint() { return float(symmetric_zero_point<Format::BITS>()); }

  template <typename RightArgs>
  static METAL_FUNC float offset(
      const float scale,
      const uint scale_index,
      const uint column,
      const uint group,
      const uint groups_per_row,
      const thread RightArgs& right
  ) {
    if constexpr (RightOperand::SCHEME == GemmBPrologueKind::ScaleBiasDequant) {
      return static_cast<float>(right.biases[scale_index]);
    } else if constexpr (RightOperand::SCHEME == GemmBPrologueKind::ScaleZeroPointDequant) {
      if constexpr (Format::BITS == 8) {
        return -scale * float(right.zero_points[scale_index]);
      } else {
        const device uint8_t* zero_points_row =
            right.zero_points + column * zero_point_row_stride<Format::BITS>(groups_per_row);
        return -scale * float(decode_zero_point<Format::BITS>(zero_points_row, group));
      }
    } else {
      return 0.0f;
    }
  }
};

} // namespace gemm
} // namespace uzu
