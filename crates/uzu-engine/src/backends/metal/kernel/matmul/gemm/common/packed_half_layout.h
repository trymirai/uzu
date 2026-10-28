#pragma once

#include <metal_stdlib>
#include "../../../common/defines.h"

using namespace metal;

namespace uzu::gemm {

template <uint GROUP_SIZE>
struct PackedHalfLayout {
  static_assert(GROUP_SIZE == 32 || GROUP_SIZE == 64);
  UZU_CONST uint MMA_WIDTH = 8;
  UZU_CONST uint VALUES_PER_VECTOR = 4;
  UZU_CONST uint ROWS_PER_PAIR = 2;
  UZU_CONST uint SUMS_PER_ROW = 2;
  UZU_CONST uint W4_CODES_PER_BYTE = 2;
  UZU_CONST uint G64_K_INTERLEAVE = 2;
  UZU_CONST uint G64_K_CHUNK = G64_K_INTERLEAVE * MMA_WIDTH;
  UZU_CONST uint WEIGHT_WORD_BYTES = sizeof(uint);
  UZU_CONST uint WEIGHT_WORDS_PER_LANE = GROUP_SIZE / METAL_SIMD_SIZE;
  UZU_CONST uint STEPS = GROUP_SIZE / MMA_WIDTH;
  UZU_CONST uint SUM_WORDS = (METAL_SIMD_SIZE / MMA_WIDTH) * ROWS_PER_PAIR * SUMS_PER_ROW;
  UZU_CONST uint GROUP_WORDS = METAL_SIMD_SIZE * STEPS + SUM_WORDS;

  static METAL_FUNC uint pair_offset(uint slot, uint fm) { return (slot * MMA_WIDTH + fm) * STEPS; }
  static METAL_FUNC uint codes_offset(uint group, uint slot) {
    return group * (GROUP_SIZE / W4_CODES_PER_BYTE) + slot * (WEIGHT_WORD_BYTES * WEIGHT_WORDS_PER_LANE);
  }
  static METAL_FUNC uint sums_offset() { return METAL_SIMD_SIZE * STEPS; }
  static METAL_FUNC uint word_offset(uint k, uint row) {
    const uint chunk = (k % G64_K_CHUNK) / VALUES_PER_VECTOR;
    const uint fm =
        GROUP_SIZE == 32 ? k / VALUES_PER_VECTOR : (k / G64_K_CHUNK) * G64_K_INTERLEAVE + chunk % G64_K_INTERLEAVE;
    const uint step = GROUP_SIZE == 32 ? k % VALUES_PER_VECTOR
                                       : (chunk / G64_K_INTERLEAVE) * VALUES_PER_VECTOR + k % VALUES_PER_VECTOR;
    return pair_offset(row / ROWS_PER_PAIR, fm) + step;
  }
};

} // namespace uzu::gemm
