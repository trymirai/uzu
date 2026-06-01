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

template <typename U, int N, int bits>
inline void dequantize(
    const device uint8_t* w,
    U scale,
    U bias,
    threadgroup U* w_local
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  if (bits == 4) {
    U s0 = scale;
    U s1 = scale / static_cast<U>(16.0f);
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = s0 * (w[i] & 0x0f) + bias;
      w_local[2 * i + 1] = s1 * (w[i] & 0xf0) + bias;
    }
  } else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = scale * w[i] + bias;
    }
  }
}

template <>
inline void dequantize<bfloat, 8, 4>(
    const device uint8_t* w,
    bfloat scale,
    bfloat bias,
    threadgroup bfloat* w_local
) {
  const uint32_t packed = *reinterpret_cast<const device uint32_t*>(w);
  // Mask the whole 32-bit word and reinterpret as uchar4 — gets all four low
  // (resp. high) nibbles in one vector op, with NO per-nibble shifts. High
  // nibbles stay shifted left by 4 (value * 16), compensated by scaling with
  // scale/16. ~12 ALU ops vs ~30 for the per-nibble shift + per-element
  // int->bfloat convert form (measured in AIR).
  const bfloat4 lo =
      bfloat4(as_type<uchar4>(packed & 0x0f0f0f0fu)) * scale + bias;
  const bfloat4 hi = bfloat4(as_type<uchar4>(packed & 0xf0f0f0f0u)) *
                         (scale * bfloat(0.0625f)) +
                     bias;

  // Interleave back to nibble order: [b0lo, b0hi, b1lo, b1hi, ...].
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
