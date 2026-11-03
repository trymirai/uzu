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

template <uint ORIGIN>
METAL_FUNC float magic_code_float(uint code) {
  return as_type<float>(code + W8_MAGIC_FLOAT_OFFSET - ORIGIN);
}

METAL_FUNC float magic_dequant_bias(float scale) { return -12582912.0f * scale; }

// fma(code, scale, bias) in f32, rounded once to U, stored in pairs.
template <typename U, int N, int bits>
inline void dequantize(
    const device uint8_t* w,
    float scale,
    float bias,
    threadgroup U* w_local,
    const bool signed_codes
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");
  static_assert(bits == 8 || N % 2 == 0, "int4 values are unpacked in pairs");
  threadgroup vec<U, 2>* pairs = reinterpret_cast<threadgroup vec<U, 2>*>(w_local);
  if constexpr (bits == 4) {
    const uint sign_flip = signed_codes ? 0x88u : 0u;
    for (int i = 0; i < (N / 2); i++) {
      const uint code = uint(w[i]) ^ sign_flip;
      pairs[i] = vec<U, 2>(vec<float, 2>(fma(float(code & 0x0fu), scale, bias), fma(float(code >> 4), scale, bias)));
    }
  } else {
    const uint sign_flip = signed_codes ? 0x80u : 0u;
    for (int i = 0; i < (N / 2); i++) {
      pairs[i] = vec<U, 2>(vec<float, 2>(
          fma(float(uint(w[2 * i]) ^ sign_flip), scale, bias),
          fma(float(uint(w[2 * i + 1]) ^ sign_flip), scale, bias)
      ));
    }
    if constexpr (N % 2 == 1) {
      w_local[N - 1] = U(fma(float(uint(w[N - 1]) ^ sign_flip), scale, bias));
    }
  }
}

} // namespace gemm
} // namespace uzu
