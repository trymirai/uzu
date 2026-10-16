#pragma once

#include "../../common/quant_unpack.h"
#include "../../../common/conversion.h"

namespace uzu {
namespace gemm {

template <uint BITS>
struct QuantChunk {
  UZU_CONST uint VALUES = 8;
  UZU_CONST uint BITS_PER_BYTE = 8;
  UZU_CONST uint WORD_BITS = 32;
  UZU_CONST uint WORD_LANES = 4;
  UZU_CONST uint WORD_BYTES = WORD_BITS / BITS_PER_BYTE;
  UZU_CONST uint VECTOR_BYTES = WORD_BYTES * WORD_LANES;

  template <uint WORDS>
  static METAL_FUNC void decode(
      const thread uint4 (&words)[WORDS],
      uint chunk,
      thread float (&values)[VALUES],
      float origin,
      bool signed_codes
  ) {
    const uint word = chunk * (BITS == 4 ? 1u : 2u);
    const uint word0 = words[word / 4u][word % 4u];
    const uint word1 = BITS == 4 ? 0u : words[(word + 1u) / 4u][(word + 1u) % 4u];
    decode_words(word0, word1, values, origin, signed_codes);
  }

  template <bool ALIGNED, typename AT>
  static METAL_FUNC void load(
      const device AT* source,
      thread float (&values)[VALUES],
      uint offset,
      uint extent,
      uint row
  ) {
    static_assert(
        metal::is_same_v<AT, bfloat> || metal::is_same_v<AT, float>,
        "GEMV quantized input must be bfloat or float; half cannot use bfloat widening"
    );
    constexpr bool EXPANDS_BF16 = metal::is_same_v<AT, bfloat>;
    if constexpr (ALIGNED) {
      load_values<false, EXPANDS_BF16>(source, values, VALUES);
    } else {
      const uint valid = offset < extent ? min(extent - offset, VALUES) : 0u;
      const bool source_aligned = !EXPANDS_BF16 || ((row * extent + offset) & 7u) == 0;
      if (valid == VALUES && source_aligned) {
        load_values<false, EXPANDS_BF16>(source, values, VALUES);
      } else {
        load_values<true, EXPANDS_BF16>(source, values, valid);
      }
    }
  }

private:
  static METAL_FUNC void decode_words(
      uint word0,
      uint word1,
      thread float (&values)[VALUES],
      float origin,
      bool signed_codes
  ) {
    constexpr uint CODES_PER_WORD = WORD_BITS / BITS;
    if constexpr (BITS == 4) {
      word0 ^= signed_codes ? W4_SIGN_MASK : 0u;
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < VALUES; i++) {
        values[i] = float_with_offset(extract_bits(word0, BITS * i, BITS)) - origin;
      }
    } else {
      const uint mask = signed_codes ? W8_SIGN_MASK : 0u;
      word0 ^= mask;
      word1 ^= mask;
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < CODES_PER_WORD; i++) {
        values[i] = float_with_offset(extract_bits(word0, BITS * i, BITS)) - origin;
        values[CODES_PER_WORD + i] = float_with_offset(extract_bits(word1, BITS * i, BITS)) - origin;
      }
    }
  }

  template <bool BOUNDED, bool EXPANDS_BF16, typename AT>
  static METAL_FUNC void load_values(const device AT* source, thread float (&values)[VALUES], uint valid) {
    if constexpr (!BOUNDED && EXPANDS_BF16) {
      const uint4 packed = *reinterpret_cast<const device uint4*>(source);
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < WORD_LANES; i++) {
        const float2 widened = unpack_bfloat(packed[i]);
        values[2 * i] = widened.x;
        values[2 * i + 1] = widened.y;
      }
    } else {
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < VALUES; i++) {
        float value = 0.0f;
        if constexpr (BOUNDED) {
          if (i < valid) {
            value = float(source[i]);
          }
        } else {
          value = float(source[i]);
        }
        values[i] = value;
      }
    }
  }
};

} // namespace gemm
} // namespace uzu
