#pragma once

#include <metal_stdlib>

namespace uzu {

template <uint BITS>
inline uint read_packed(uchar packed, uint index) {
  static_assert(BITS == 2 || BITS == 4, "Only 2-bit and 4-bit packed values are supported");
  return (uint(packed) >> (BITS * (index % (8u / BITS)))) & ((1u << BITS) - 1u);
}

template <uint BITS>
inline uint read_packed(const device uchar* packed, uint index) {
  return read_packed<BITS>(packed[index / (8u / BITS)], index);
}

} // namespace uzu
