#pragma once

#include <metal_stdlib>

#include "../../common/defines.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <typename U>
METAL_FUNC U uint_to_fp(uint32_t x) {
  return static_cast<U>(as_type<float>(x | 0x4B000000u) - 8388608.0f);
}

template <>
METAL_FUNC bfloat uint_to_fp<bfloat>(uint32_t x) {
  return as_type<bfloat>(uint16_t(x | 0x4300u)) - bfloat(128.0f);
}

template <int BITS>
METAL_FUNC float4 _uint4_to_fp4_float(uint4 n) {
  static_assert(BITS > 0 && BITS <= 23, "BITS must fit in float23 mantissa");
  constexpr uint mask = (1u << BITS) - 1u;
  n &= uint4(mask);
  return as_type<float4>(n | uint4(0x4B000000u)) - float4(8388608.0f);
}

template <typename U, int BITS>
METAL_FUNC vec<U, 4> uint4_to_fp4(uint4 n);

template <>
METAL_FUNC float4 uint4_to_fp4<float, 4>(uint4 n) {
  return _uint4_to_fp4_float<4>(n);
}

template <>
METAL_FUNC float4 uint4_to_fp4<float, 8>(uint4 n) {
  return _uint4_to_fp4_float<8>(n);
}

template <>
METAL_FUNC bfloat4 uint4_to_fp4<bfloat, 4>(uint4 n) {
  return bfloat4(_uint4_to_fp4_float<4>(n));
}

template <>
METAL_FUNC bfloat4 uint4_to_fp4<bfloat, 8>(uint4 n) {
  return bfloat4(_uint4_to_fp4_float<8>(n));
}

template <ushort BITS>
METAL_FUNC constexpr uint symmetric_zero_point() {
  return 1u << (BITS - 1);
}

template <ushort BITS, typename Int>
METAL_FUNC constexpr Int zero_point_row_stride(Int groups_per_row) {
  return (BITS == 4) ? (groups_per_row + Int(1)) / Int(2) : groups_per_row;
}

template <ushort BITS>
METAL_FUNC uint decode_zero_point(const device uint8_t* zero_points_row, uint group_index) {
  static_assert(BITS == 4 || BITS == 8, "Only int4 and int8 zero points supported");
  if constexpr (BITS == 4) {
    const uint packed = uint(zero_points_row[group_index >> 1]);
    return (packed >> ((group_index & 1u) * 4u)) & 0x0Fu;
  } else {
    return uint(zero_points_row[group_index]);
  }
}

METAL_FUNC char4 unpack_signed_nibbles_to_int8(uint packed) {
  uint spread = (packed | (packed << 8)) & 0x00FF00FFu;
  spread = (spread | (spread << 4)) & 0x0F0F0F0Fu;
  constexpr uint sign_bits = symmetric_zero_point<4>() * 0x01010101u;
  return as_type<char4>(spread ^ sign_bits) - char4(char(symmetric_zero_point<4>()));
}

template <typename U, int N, int bits>
inline void dequantize(const device uint8_t* w, U scale, U bias, threadgroup U* w_local, const bool signed_codes) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  if constexpr (bits == 4) {
    const uint8_t sign_flip_mask = signed_codes ? 0x88u : 0u;
    U s0 = scale;
    U s1 = scale / static_cast<U>(16.0f);
    for (int i = 0; i < (N / 2); i++) {
      const uint8_t word = w[i] ^ uint8_t(sign_flip_mask);
      w_local[2 * i] = s0 * (word & 0x0f) + bias;
      w_local[2 * i + 1] = s1 * (word & 0xf0) + bias;
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

template <>
inline void dequantize<bfloat, 8, 4>(
    const device uint8_t* w,
    bfloat scale,
    bfloat bias,
    threadgroup bfloat* w_local,
    const bool signed_codes
) {
  const uint32_t packed_mask = signed_codes ? 0x88888888u : 0u;
  const uint32_t packed = (*reinterpret_cast<const device uint32_t*>(w)) ^ packed_mask;
  const bfloat4 lo = bfloat4(as_type<uchar4>(packed & 0x0f0f0f0fu)) * scale + bias;
  const bfloat4 hi = bfloat4(as_type<uchar4>(packed & 0xf0f0f0f0u)) * (scale * bfloat(0.0625f)) + bias;

  w_local[0] = lo.x;
  w_local[1] = hi.x;
  w_local[2] = lo.y;
  w_local[3] = hi.y;
  w_local[4] = lo.z;
  w_local[5] = hi.z;
  w_local[6] = lo.w;
  w_local[7] = hi.w;
}

} // namespace gemm
} // namespace uzu
