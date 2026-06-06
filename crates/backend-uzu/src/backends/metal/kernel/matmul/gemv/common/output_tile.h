#pragma once

#include "gemv_common.h"

namespace uzu {
namespace gemm {

template <uint K_SPLIT, uint NUM_SIMDGROUPS>
struct OutputTile {
  uint out_row;
  uint row_group;
  uint k_slice;
  bool writer;

  static METAL_FUNC OutputTile make(uint out_block_idx, uint simd_group, uint out_vec_size) {
    constexpr uint rows_per_threadgroup = (NUM_SIMDGROUPS / K_SPLIT) * RESULTS_PER_SIMDGROUP;
    const uint row_group = simd_group / K_SPLIT;
    const uint k_slice = simd_group % K_SPLIT;
    uint out_row = out_block_idx * rows_per_threadgroup + row_group * RESULTS_PER_SIMDGROUP;
    out_row = out_vec_size > RESULTS_PER_SIMDGROUP ? min(out_row, out_vec_size - RESULTS_PER_SIMDGROUP) : 0u;
    OutputTile tile;
    tile.out_row = out_row;
    tile.row_group = row_group;
    tile.k_slice = k_slice;
    tile.writer = (K_SPLIT == 1) || (k_slice == 0);
    return tile;
  }
};

} // namespace gemm
} // namespace uzu
