#pragma once

#include <metal_stdlib>

namespace uzu {
namespace gemm {

template <int bits, int word_size_bits = 8>
METAL_FUNC constexpr short get_pack_factor() {
  return word_size_bits / bits;
}

template <int bits, int word_size_bits = 8>
METAL_FUNC constexpr short get_bytes_per_pack() {
  return word_size_bits / 8;
}

} // namespace gemm
} // namespace uzu
