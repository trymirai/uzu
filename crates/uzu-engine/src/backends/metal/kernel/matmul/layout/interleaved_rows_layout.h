#pragma once

#include "../../common/defines.h"
#include "../../generated/matmul.h"

namespace uzu {
namespace gemm {

// 32-row strips, word j of rows ordered 0, 8, 16, 24 | 1, 9, 17, 25 | ... so a GEMM lane's 4 rows are one uint4.
struct InterleavedRowsLayout {
  UZU_CONST uint STRIP_ROWS = uzu::matmul::INTERLEAVED_ROWS_STRIP_ROWS;
  UZU_CONST uint LANE_ROWS = uzu::matmul::INTERLEAVED_ROWS_LANE_ROWS;
  UZU_CONST uint ROWS_PER_LANE = STRIP_ROWS / LANE_ROWS;

  static METAL_FUNC size_t word_index(uint row, uint word, uint row_words) {
    return size_t(row / STRIP_ROWS) * STRIP_ROWS * row_words + word * STRIP_ROWS + row % LANE_ROWS * ROWS_PER_LANE +
           row / LANE_ROWS % ROWS_PER_LANE;
  }

  static METAL_FUNC uint row_major_word_in_strip(uint index, uint row_words) {
    const uint slot = index % STRIP_ROWS;
    return (slot % ROWS_PER_LANE * LANE_ROWS + slot / ROWS_PER_LANE) * row_words + index / STRIP_ROWS;
  }

  static METAL_FUNC int byte_offset(int byte) {
    const int word_bytes = sizeof(uint);
    return byte / word_bytes * int(STRIP_ROWS) * word_bytes + byte % word_bytes;
  }

  static METAL_FUNC uint4 lane_run(const device uint8_t* codes, uint row, uint word, uint row_words) {
    return *reinterpret_cast<const device uint4*>(
        reinterpret_cast<const device uint*>(codes) + word_index(row, word, row_words)
    );
  }
};

} // namespace gemm
} // namespace uzu
