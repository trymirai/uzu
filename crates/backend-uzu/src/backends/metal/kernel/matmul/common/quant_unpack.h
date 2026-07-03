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

} // namespace gemm
} // namespace uzu
