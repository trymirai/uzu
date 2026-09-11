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
  bool signed_codes;

public:
  static METAL_FUNC uint group_count(const thread GemvParams& params, uint group_size) {
    return (params.in_vec_size + group_size - 1) / group_size;
  }

  METAL_FUNC void load(
      uint group_index,
      uint group_count,
      const thread uint (&weight_rows)[Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params
  ) thread {
    signed_codes = params.signed_codes;
    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      const uint row = weight_rows[R];
      scale[R] = float(ops.scales[row * group_count + group_index]);
      if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant) {
        constexpr uint ZERO_POINTS_PER_BYTE = QuantChunk<BITS>::BITS_PER_BYTE / BITS;
        const uint byte_index = group_index / ZERO_POINTS_PER_BYTE;
        const uint8_t packed = ops.zero_points[row * zero_point_row_stride<BITS>(group_count) + byte_index];
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
      thread float (&values)[QuantChunk<BITS>::VALUES]
  ) const thread {
    QuantChunk<BITS>::decode(words, chunk, values, origin[row], signed_codes);
  }

  METAL_FUNC void fold(
      thread float (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread float (&partial)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread float (&input_sum)[Tile::INPUT_ROWS]
  ) const thread {
    Tile::for_each_input_row([&](auto input_index) UZU_ALWAYS_INLINE {
      constexpr uint I = decltype(input_index)::value;
      Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
        constexpr uint R = decltype(output_index)::value;
        result[I][R] = finish(result[I][R], partial[I][R], input_sum[I], R);
      });
    });
  }

private:
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
