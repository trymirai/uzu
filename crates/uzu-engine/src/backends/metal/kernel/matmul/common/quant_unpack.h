#pragma once

#include <metal_stdlib>

#include "../../common/defines.h"
#include "../../common/packed.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <ushort BITS>
METAL_FUNC constexpr uint symmetric_zero_point() {
  return 1u << (BITS - 1);
}

template <ushort BITS>
METAL_FUNC uint decode_zero_point(uint8_t packed, uint group_index) {
  static_assert(BITS == 4, "zero points are 4-bit");
  return read_packed<4>(packed, group_index);
}

template <ushort BITS>
METAL_FUNC uint decode_zero_point(const device uint8_t* zero_points_row, uint group_index) {
  static_assert(BITS == 4, "zero points are 4-bit");
  return read_packed<4>(zero_points_row, group_index);
}

UZU_CONST ushort W4_BITS = 4;
UZU_CONST uint W4_SIGN_MASK = symmetric_zero_point<W4_BITS>() * 0x11111111u;
UZU_CONST uint W8_SIGN_MASK = symmetric_zero_point<8>() * 0x01010101u;
UZU_CONST uint W4_NIBBLE_MASK = 0x0F0F0F0Fu;
UZU_CONST uint ZERO_POINTS_PER_BYTE = 2;
UZU_CONST uint W8_MAGIC_FLOAT_OFFSET = 0x4B400000u;

UZU_CONST uint W4_PAIR_HALF_BITS = 16;
UZU_CONST uint W4_PAIR_MANTISSA_BITS = 10;
UZU_CONST uint W4_PAIR_EXPONENT_BIAS = 15;

static METAL_FUNC constexpr uint w4_pair_offset(uint component) {
  constexpr uint COMPONENTS = W4_PAIR_HALF_BITS / W4_BITS;
  return W4_BITS * (component < COMPONENTS ? component : COMPONENTS - 1);
}

static METAL_FUNC constexpr uint w4_pair_position(uint component) {
  const uint offset = w4_pair_offset(component);
  return offset + W4_BITS <= W4_PAIR_MANTISSA_BITS ? offset : offset - (W4_PAIR_HALF_BITS - W4_PAIR_MANTISSA_BITS);
}

template <uint COMPONENT>
static METAL_FUNC half2 w4_pair_at(uint word) {
  constexpr uint offset = w4_pair_offset(COMPONENT);
  constexpr uint position = w4_pair_position(COMPONENT);
  constexpr uint mask = ((1u << W4_BITS) - 1) << position;
  constexpr uint exponent = (W4_PAIR_EXPONENT_BIAS + W4_PAIR_MANTISSA_BITS - position) << W4_PAIR_MANTISSA_BITS;
  return as_type<half2>(
      ((word >> (offset - position)) & (mask | (mask << W4_PAIR_HALF_BITS))) |
      (exponent | (exponent << W4_PAIR_HALF_BITS))
  );
}

static METAL_FUNC constexpr float w4_pair_bias_value(uint component) {
  return float(1u << (W4_PAIR_MANTISSA_BITS - w4_pair_position(component)));
}

static METAL_FUNC half2 w4_pair(uint word, ushort component) {
  switch (component) {
  case 0:
    return w4_pair_at<0>(word);
  case 1:
    return w4_pair_at<1>(word);
  case 2:
    return w4_pair_at<2>(word);
  default:
    return w4_pair_at<3>(word);
  }
}

UZU_CONST float W4_PAIR_BIAS_0 = w4_pair_bias_value(0);
UZU_CONST float W4_PAIR_BIAS_1 = w4_pair_bias_value(1);
UZU_CONST float W4_PAIR_BIAS_2 = w4_pair_bias_value(2);
UZU_CONST float W4_PAIR_BIAS_3 = w4_pair_bias_value(3);

static METAL_FUNC float w4_pair_bias(uint component) {
  return component == 0   ? W4_PAIR_BIAS_0
         : component == 1 ? W4_PAIR_BIAS_1
         : component == 2 ? W4_PAIR_BIAS_2
                          : W4_PAIR_BIAS_3;
}

template <uint ORIGIN>
METAL_FUNC float magic_code_float(uint code) {
  return as_type<float>(code + W8_MAGIC_FLOAT_OFFSET - ORIGIN);
}

METAL_FUNC float magic_dequant_bias(float scale) { return -12582912.0f * scale; }

template <typename U, int N, int bits>
inline void dequantize(const device uint8_t* w, U scale, U bias, threadgroup U* w_local, const bool signed_codes) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  if constexpr (bits == 4) {
    U s0 = scale;
    U s1 = scale / static_cast<U>(16.0f);
    // Keeping literal masks in each arm lets the compiler vectorize the unpack.
    if (signed_codes) {
      for (int i = 0; i < (N / 2); i++) {
        const uint8_t word = w[i] ^ uint8_t(0x88u);
        w_local[2 * i] = s0 * (word & 0x0f) + bias;
        w_local[2 * i + 1] = s1 * (word & 0xf0) + bias;
      }
    } else {
      for (int i = 0; i < (N / 2); i++) {
        w_local[2 * i] = s0 * (w[i] & 0x0f) + bias;
        w_local[2 * i + 1] = s1 * (w[i] & 0xf0) + bias;
      }
    }
  } else if constexpr (bits == 8) {
    if (signed_codes) {
      const device int8_t* signed_weights = reinterpret_cast<const device int8_t*>(w);
      const U adjusted_bias = bias + scale * U(128);
      for (int i = 0; i < N; i++) {
        w_local[i] = scale * U(signed_weights[i]) + adjusted_bias;
      }
    } else {
      for (int i = 0; i < N; i++) {
        w_local[i] = scale * U(w[i]) + bias;
      }
    }
  }
}

} // namespace gemm
} // namespace uzu
