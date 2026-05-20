#pragma once

#include <metal_stdlib>

#include "../../common/defines.h"

using namespace metal;

namespace uzu {
namespace gemm {

// Avoids air.convert (SFU) for int → float, which is slower on Apple GPU.
template <typename U>
METAL_FUNC U uint_to_fp(uint32_t x) {
  return static_cast<U>(as_type<float>(x | 0x4B000000u) - 8388608.0f);
}

template <>
METAL_FUNC bfloat uint_to_fp<bfloat>(uint32_t x) {
  return as_type<bfloat>(uint16_t(x | 0x4300u)) - bfloat(128.0f);
}

// Unpack 4 lanes of `BITS`-wide packed uints (low bits of each uint) into
// `vec<U, 4>` of floats. `BITS` must fit in the float23 mantissa.
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
METAL_FUNC half4 uint4_to_fp4<half, 4>(uint4 n) {
  return half4(_uint4_to_fp4_float<4>(n));
}

template <>
METAL_FUNC half4 uint4_to_fp4<half, 8>(uint4 n) {
  return half4(_uint4_to_fp4_float<8>(n));
}

template <>
METAL_FUNC bfloat4 uint4_to_fp4<bfloat, 4>(uint4 n) {
  return bfloat4(_uint4_to_fp4_float<4>(n));
}

template <>
METAL_FUNC bfloat4 uint4_to_fp4<bfloat, 8>(uint4 n) {
  return bfloat4(_uint4_to_fp4_float<8>(n));
}

// Dequantize `N` packed weights from device → threadgroup memory.
// The generic implementation handles bits=4/8; the bfloat × N=8 × bits=4 case
// has an unrolled specialization below that fuses four nibble extracts into
// one bfloat4 multiply-add pair.
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
  const device uint32_t* w_ptr = reinterpret_cast<const device uint32_t*>(w);
  uint32_t packed = *w_ptr;

  bfloat4 v0, v1;

  // Low 4 nibbles
  v0.x = static_cast<bfloat>(packed & 0xF);
  v0.y = static_cast<bfloat>((packed >> 4) & 0xF);
  v0.z = static_cast<bfloat>((packed >> 8) & 0xF);
  v0.w = static_cast<bfloat>((packed >> 12) & 0xF);

  // High 4 nibbles
  v1.x = static_cast<bfloat>((packed >> 16) & 0xF);
  v1.y = static_cast<bfloat>((packed >> 20) & 0xF);
  v1.z = static_cast<bfloat>((packed >> 24) & 0xF);
  v1.w = static_cast<bfloat>((packed >> 28) & 0xF);

  v0 = v0 * scale + bias;
  v1 = v1 * scale + bias;

  threadgroup bfloat4* out_ptr = reinterpret_cast<threadgroup bfloat4*>(w_local);
  out_ptr[0] = v0;
  out_ptr[1] = v1;
}

} // namespace gemm
} // namespace uzu
