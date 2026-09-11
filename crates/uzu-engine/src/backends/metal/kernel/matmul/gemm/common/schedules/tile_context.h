#pragma once

#include <metal_stdlib>

using namespace metal;

namespace uzu {
namespace gemm {
namespace schedules {

struct TileContext {
  short simdgroup_limit_m;
  short simdgroup_limit_n;
  size_t block_col;
  ushort tile_col_offset;
  ushort tile_block_cols;
  uint k_offset;
  uint abs_row_base;

  METAL_FUNC uint absolute_column_base() const { return uint(block_col) + uint(tile_col_offset); }
};

} // namespace schedules
} // namespace gemm
} // namespace uzu
