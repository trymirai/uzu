#pragma once

#include <metal_stdlib>

namespace uzu {
namespace gemm {

// Layout constants for bit-packed weights. `pack_factor` is values per pack;
// `bytes_per_pack` is the byte stride of one pack. Together they encode the
// minimal-overhead packing for both power-of-two and non-power-of-two bit
// widths (e.g. bits=5 → 8 values in 5 bytes = 40 bits, perfect packing).
template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

} // namespace gemm
} // namespace uzu
