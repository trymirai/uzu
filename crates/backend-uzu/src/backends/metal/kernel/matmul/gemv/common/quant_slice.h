#pragma once

#include "arguments.h"
#include "quant_chunk.h"
#include "quant_metadata.h"
#include "tile.h"

namespace uzu {
namespace gemm {

template <
    typename Tile,
    typename AT,
    typename BT,
    typename DT,
    GemmBPrologueKind B_PROLOGUE,
    uint GROUP_SIZE,
    uint BITS,
    bool INPUT_ALIGNED,
    bool FULL_TILE>
struct QuantSlice {
  using U = float;

private:
  UZU_CONST uint VALUES_PER_LANE = GROUP_SIZE / Tile::GROUP_LANES;
  UZU_CONST uint CHUNK_VALUES = QuantChunk<BITS>::VALUES;
  UZU_CONST uint MAX_SLICE_VALUES = 32;
  UZU_CONST uint SLICE_VALUES = VALUES_PER_LANE > MAX_SLICE_VALUES ? MAX_SLICE_VALUES : VALUES_PER_LANE;

public:
  UZU_CONST uint SLICES_PER_LANE = VALUES_PER_LANE / SLICE_VALUES;

  struct Position {
    uint group;
    uint slice;

    METAL_FUNC bool valid(uint groups) const thread { return group < groups; }
  };

private:
  UZU_CONST uint SLICE_BYTES = SLICE_VALUES * BITS / QuantChunk<BITS>::BITS_PER_BYTE;
  UZU_CONST uint SLICE_WORDS = (SLICE_BYTES + QuantChunk<BITS>::VECTOR_BYTES - 1) / QuantChunk<BITS>::VECTOR_BYTES;
  UZU_CONST uint CHUNKS_PER_SLICE = SLICE_VALUES / CHUNK_VALUES;

  static_assert(GROUP_SIZE % Tile::GROUP_LANES == 0, "group lanes must divide the quantization group");
  static_assert(VALUES_PER_LANE % CHUNK_VALUES == 0, "group lane slices must contain complete chunks");
  static_assert(SLICE_VALUES <= MAX_SLICE_VALUES, "QMV slice exceeds its staged-value limit");
  static_assert(VALUES_PER_LANE % SLICE_VALUES == 0, "a lane loads complete group slices");

  uint4 weights[Tile::ROWS_PER_LANE][SLICE_WORDS];

public:
  METAL_FUNC void load_weights(
      const thread Position& position,
      const device uint8_t* weights_base,
      const thread uint (&weight_row_indices)[Tile::ROWS_PER_LANE],
      uint row_stride,
      uint group_offset
  ) thread {
    const uint k = position.group * GROUP_SIZE + group_offset + position.slice * SLICE_VALUES;
    const uint offset_bytes = k * BITS / QuantChunk<BITS>::BITS_PER_BYTE;
    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      const uint row = weight_row_indices[R];
      const device uint8_t* source = weights_base + row * row_stride + offset_bytes;
      if constexpr (INPUT_ALIGNED) {
        load_words<false>(weights[R], source);
      } else {
        const uint valid_bytes = offset_bytes < row_stride ? min(row_stride - offset_bytes, SLICE_BYTES) : 0u;
        constexpr uint LOAD_ALIGNMENT =
            SLICE_BYTES < QuantChunk<BITS>::VECTOR_BYTES ? SLICE_BYTES : QuantChunk<BITS>::VECTOR_BYTES;
        const bool load_aligned = ((row * row_stride + offset_bytes) & (LOAD_ALIGNMENT - 1)) == 0;
        if (valid_bytes == SLICE_BYTES && load_aligned) {
          load_words<false>(weights[R], source);
        } else {
          load_words<true>(weights[R], source, valid_bytes);
        }
      }
    });
  }

  METAL_FUNC void accumulate(
      thread U (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread Position& position,
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread OutputTile<Tile, FULL_TILE>& tile,
      uint group_offset,
      uint batch_remaining,
      const thread QuantMetadata<Tile, AT, BT, DT, B_PROLOGUE, BITS>& metadata
  ) const thread {
    float partial[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE] = {{0}};
    float input_sum[Tile::INPUT_ROWS] = {0};
    METAL_PRAGMA_UNROLL
    for (uint chunk = 0; chunk < CHUNKS_PER_SLICE; chunk++) {
      float weight_values[Tile::ROWS_PER_LANE][CHUNK_VALUES];
      Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
        constexpr uint R = decltype(output_index)::value;
        metadata.decode(weights[R], chunk, R, params.signed_codes, weight_values[R]);
      });
      Tile::for_each_input_row([&](auto input_index) UZU_ALWAYS_INLINE {
        constexpr uint I = decltype(input_index)::value;
        const uint source_row = FULL_TILE ? I : min(I, batch_remaining - 1);
        const uint k =
            position.group * GROUP_SIZE + group_offset + position.slice * SLICE_VALUES + chunk * CHUNK_VALUES;
        const uint input_row = tile.input_row + source_row;
        const device AT* input = ops.a + input_row * params.in_vec_size + k;
        float input_values[CHUNK_VALUES];
        QuantChunk<BITS>::template load<INPUT_ALIGNED>(input, input_values, k, params.in_vec_size, input_row);
        if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
          METAL_PRAGMA_UNROLL
          for (uint i = 0; i < CHUNK_VALUES; i++) {
            input_sum[I] += input_values[i];
          }
        }
        Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
          constexpr uint R = decltype(output_index)::value;
          METAL_PRAGMA_UNROLL
          for (uint i = 0; i < CHUNK_VALUES; i++) {
            partial[I][R] = fma(input_values[i], weight_values[R][i], partial[I][R]);
          }
        });
      });
    }
    Tile::for_each_input_row([&](auto input_index) UZU_ALWAYS_INLINE {
      constexpr uint I = decltype(input_index)::value;
      Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
        constexpr uint R = decltype(output_index)::value;
        result[I][R] = metadata.finish(result[I][R], partial[I][R], input_sum[I], R);
      });
    });
  }

private:
  template <bool BOUNDED>
  static METAL_FUNC void load_words(
      thread uint4 (&words)[SLICE_WORDS],
      const device uint8_t* source,
      uint valid_bytes = SLICE_BYTES
  ) {
    if constexpr (!BOUNDED && SLICE_BYTES == QuantChunk<BITS>::WORD_BYTES) {
      const uint packed = *reinterpret_cast<const device uint*>(source);
      words[0] = uint4(packed, 0u, 0u, 0u);
    } else if constexpr (!BOUNDED && SLICE_BYTES == 2 * QuantChunk<BITS>::WORD_BYTES) {
      const uint2 packed = *reinterpret_cast<const device uint2*>(source);
      words[0] = uint4(packed[0], packed[1], 0u, 0u);
    } else if constexpr (!BOUNDED) {
      static_assert(SLICE_BYTES % QuantChunk<BITS>::VECTOR_BYTES == 0, "large slices contain complete uint4 words");
      const device uint4* packed = reinterpret_cast<const device uint4*>(source);
      METAL_PRAGMA_UNROLL
      for (uint word = 0; word < SLICE_WORDS; word++) {
        words[word] = packed[word];
      }
    } else {
      METAL_PRAGMA_UNROLL
      for (uint word = 0; word < SLICE_WORDS; word++) {
        uint4 packed = uint4(0u);
        METAL_PRAGMA_UNROLL
        for (uint component = 0; component < QuantChunk<BITS>::WORD_LANES; component++) {
          uint value = 0u;
          METAL_PRAGMA_UNROLL
          for (uint byte = 0; byte < QuantChunk<BITS>::WORD_BYTES; byte++) {
            const uint index = QuantChunk<BITS>::VECTOR_BYTES * word + QuantChunk<BITS>::WORD_BYTES * component + byte;
            if (index < SLICE_BYTES && index < valid_bytes) {
              value |= uint(source[index]) << (QuantChunk<BITS>::BITS_PER_BYTE * byte);
            }
          }
          packed[component] = value;
        }
        words[word] = packed;
      }
    }
  }
};

} // namespace gemm
} // namespace uzu
