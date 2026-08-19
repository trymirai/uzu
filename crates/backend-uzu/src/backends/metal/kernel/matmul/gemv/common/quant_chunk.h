#pragma once

#include "../../common/quant_unpack.h"

namespace uzu {
namespace gemm {

template <uint BITS>
struct QuantChunk {
  UZU_CONST uint VALUES = 8;
  UZU_CONST uint FLOAT_MANTISSA_BITS = 23;
  UZU_CONST uint FLOAT_EXPONENT_BIAS = 127;
  UZU_CONST uint MANTISSA_BASE_BITS = (FLOAT_EXPONENT_BIAS + FLOAT_MANTISSA_BITS) << FLOAT_MANTISSA_BITS;
  UZU_CONST float MANTISSA_BASE = float(1u << FLOAT_MANTISSA_BITS);
  UZU_CONST uint BITS_PER_BYTE = 8;
  UZU_CONST uint WORD_BITS = 32;
  UZU_CONST uint WORD_LANES = 4;
  UZU_CONST uint WORD_BYTES = WORD_BITS / BITS_PER_BYTE;
  UZU_CONST uint VECTOR_BYTES = WORD_BYTES * WORD_LANES;
  UZU_CONST uint BF16_BITS = 16;
  UZU_CONST uint BF16_VALUES_PER_WORD = WORD_BITS / BF16_BITS;

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
    if constexpr (ALIGNED) {
      load_values<false>(source, values, VALUES);
    } else {
      const uint valid = offset < extent ? min(extent - offset, VALUES) : 0u;
      const bool source_aligned = sizeof(AT) != 2 || ((row * extent + offset) & 7u) == 0;
      if (valid == VALUES && source_aligned) {
        load_values<false>(source, values, VALUES);
      } else {
        load_values<true>(source, values, valid);
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
    constexpr uint W4_SIGN_BITS = 0x88888888u;
    constexpr uint W8_SIGN_BITS = 0x80808080u;
    if constexpr (BITS == 4) {
      word0 ^= signed_codes ? W4_SIGN_BITS : 0u;
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < VALUES; i++) {
        values[i] = as_type<float>(extract_bits(word0, BITS * i, BITS) | MANTISSA_BASE_BITS) - origin;
      }
    } else {
      const uint mask = signed_codes ? W8_SIGN_BITS : 0u;
      word0 ^= mask;
      word1 ^= mask;
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < CODES_PER_WORD; i++) {
        values[i] = as_type<float>(extract_bits(word0, BITS * i, BITS) | MANTISSA_BASE_BITS) - origin;
        values[CODES_PER_WORD + i] = as_type<float>(extract_bits(word1, BITS * i, BITS) | MANTISSA_BASE_BITS) - origin;
      }
    }
  }

  template <bool BOUNDED, typename AT>
  static METAL_FUNC void load_values(const device AT* source, thread float (&values)[VALUES], uint valid) {
    if constexpr (!BOUNDED && sizeof(AT) == 2) {
      constexpr uint BF16_UPPER_BITS = 0xffff0000u;
      const uint4 packed = *reinterpret_cast<const device uint4*>(source);
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < WORD_LANES; i++) {
        values[BF16_VALUES_PER_WORD * i] = as_type<float>(packed[i] << BF16_BITS);
        values[BF16_VALUES_PER_WORD * i + 1] = as_type<float>(packed[i] & BF16_UPPER_BITS);
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
