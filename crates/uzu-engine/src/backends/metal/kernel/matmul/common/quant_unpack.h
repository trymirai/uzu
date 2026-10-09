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
UZU_CONST uint W4_NIBBLE_MASK = 0x0F0F0F0Fu;

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
