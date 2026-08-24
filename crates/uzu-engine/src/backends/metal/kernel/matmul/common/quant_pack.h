#pragma once

#include <metal_stdlib>

namespace uzu {
namespace gemm {

template <int bits, int word_size_bits = 8>
METAL_FUNC constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : word_size_bits / bits);
}

template <int bits, int word_size_bits = 8>
METAL_FUNC constexpr short get_bytes_per_pack() {
  constexpr int is_power_of_two_bits = (bits & (bits - 1)) == 0;
  return is_power_of_two_bits ? (word_size_bits / 8) : (bits == 5 ? 5 : 3);
}

} // namespace gemm
} // namespace uzu
