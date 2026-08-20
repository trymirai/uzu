#pragma once

#include "arguments.h"
#include "quant_chunk.h"
#include "tile.h"

namespace uzu {
namespace gemm {

template <typename Tile, typename AT, typename BT, typename DT, GemmBPrologueKind B_PROLOGUE, uint BITS>
struct QuantMetadata {
private:
  float scale[Tile::ROWS_PER_LANE];
  float origin[Tile::ROWS_PER_LANE];
  float bias[Tile::ROWS_PER_LANE];

public:
  METAL_FUNC void load(
      uint group_index,
      uint group_count,
      const thread uint (&weight_rows)[Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops
  ) thread {
    const uint zp_stride = zero_point_row_stride<BITS>(group_count);
    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      const uint row = weight_rows[R];
      scale[R] = float(ops.scales[row * group_count + group_index]);
      if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant) {
        constexpr uint ZERO_POINTS_PER_BYTE = QuantChunk<BITS>::BITS_PER_BYTE / BITS;
        const uint byte_index = group_index / ZERO_POINTS_PER_BYTE;
        const uint8_t packed = ops.zero_points[row * zp_stride + byte_index];
        origin[R] = QuantChunk<BITS>::MANTISSA_BASE + float(decode_zero_point<BITS>(packed, group_index));
        bias[R] = 0.0f;
      } else if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleSymmetricDequant) {
        origin[R] = QuantChunk<BITS>::MANTISSA_BASE + float(symmetric_zero_point<BITS>());
        bias[R] = 0.0f;
      } else {
        static_assert(B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant);
        origin[R] = QuantChunk<BITS>::MANTISSA_BASE;
        bias[R] = float(ops.biases[row * group_count + group_index]);
      }
    });
  }

  template <uint WORDS>
  METAL_FUNC void decode(
      const thread uint4 (&words)[WORDS],
      uint chunk,
      uint row,
      bool signed_codes,
      thread float (&values)[QuantChunk<BITS>::VALUES]
  ) const thread {
    QuantChunk<BITS>::decode(words, chunk, values, origin[row], signed_codes);
  }

  METAL_FUNC float finish(float prior, float partial, float input_sum, uint row) const thread {
    float result = fma(scale[row], partial, prior);
    if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
      result = fma(bias[row], input_sum, result);
    }
    return result;
  }
};

} // namespace gemm
} // namespace uzu
