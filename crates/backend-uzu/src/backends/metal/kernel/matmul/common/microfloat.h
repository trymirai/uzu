#pragma once

#include "defines.h"

namespace uzu {
namespace gemm {

METAL_FUNC float decode_e2m1(uint code) {
  constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const uint magnitude_bits = as_type<uint>(magnitudes[code & 0x7u]);
  const uint sign_bit = (code & 0x8u) << 28u;
  return as_type<float>(magnitude_bits | sign_bit);
}

METAL_FUNC float decode_e8m0(uint exponent) {
  if (exponent == 0u) {
    return as_type<float>(0x00400000u);
  }
  if (exponent == 255u) {
    return as_type<float>(0x7fc00000u);
  }
  return as_type<float>(exponent << 23u);
}

METAL_FUNC float decode_mxfp4(uint code, uint exponent, float outer_scale) {
  return decode_e2m1(code) * decode_e8m0(exponent) * outer_scale;
}

} // namespace gemm
} // namespace uzu
