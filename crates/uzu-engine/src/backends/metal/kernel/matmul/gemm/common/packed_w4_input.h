#pragma once

#include <metal_stdlib>

#include "../../common/defines.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <uint GROUP_SIZE>
struct PackedW4Input {
  static_assert(GROUP_SIZE == 32 || GROUP_SIZE == 64, "packed W4 supports groups 32 and 64");
  UZU_CONST uint STEPS = GROUP_SIZE / 8;
  UZU_CONST uint GROUP_WORDS = 32 * STEPS + 16;

  static METAL_FUNC uint pair_offset(uint slot, uint fm) { return (slot * 8 + fm) * STEPS; }
  static METAL_FUNC uint sums_offset() { return 32 * STEPS; }
  static METAL_FUNC uint activation_offset(uint group, uint pair, uint member) {
    return group * GROUP_SIZE + 8 * (GROUP_SIZE / 32 * pair) + 4 * member;
  }
};

static METAL_FUNC half2 w4_pair(uint word, ushort nibble) {
  uint pair;
  switch (nibble) {
  case 0:
    pair = (word & 0x000f000fu) | 0x64006400u;
    break;
  case 1:
    pair = (word & 0x00f000f0u) | 0x54005400u;
    break;
  case 2:
    pair = ((word >> 6) & 0x003c003cu) | 0x5c005c00u;
    break;
  default:
    pair = ((word >> 6) & 0x03c003c0u) | 0x4c004c00u;
    break;
  }
  return as_type<half2>(pair);
}

static METAL_FUNC float w4_pair_bias(uint nibble) {
  return nibble == 0 ? 1024.0f : (nibble == 1 ? 64.0f : (nibble == 2 ? 256.0f : 16.0f));
}

} // namespace gemm
} // namespace uzu
