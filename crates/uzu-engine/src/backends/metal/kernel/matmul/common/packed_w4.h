#pragma once

#include <metal_stdlib>

namespace uzu {
namespace gemm {

METAL_FUNC uint packed_w4_word_offset(uint row, uint word, uint k) {
  const uint column = word & 3;
  const uint lane = (column & 1) | ((row & 3) << 1) | ((column & 2) << 2) | ((row & 4) << 2);
  return ((((row / 32) * (k / 128) + word / 16) * 4 + (row % 32) / 8) * 32 + lane) * 4 + (word % 16) / 4;
}

} // namespace gemm
} // namespace uzu
