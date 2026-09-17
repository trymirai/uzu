#pragma once

#include <metal_stdlib>

#include "../../../common/integral_constant.h"
#include "../../common/defines.h"

namespace uzu {
namespace gemm {

struct QuantPosition {
  uint group;
  uint slice;

  METAL_FUNC bool valid(uint groups) const thread { return group < groups; }
};

template <
    uint INPUT_ROW_TILE_,
    uint OUTPUT_ROW_TILE_,
    uint REDUCTION_LANES_,
    uint GROUP_LANES_,
    uint NUM_SIMDGROUPS_,
    uint K_SPLIT_>
struct GemvTile {
  // Compile-time geometry.
  UZU_CONST uint INPUT_ROWS = INPUT_ROW_TILE_;
  UZU_CONST uint OUTPUT_ROWS = OUTPUT_ROW_TILE_;
  UZU_CONST uint REDUCTION_LANES = REDUCTION_LANES_;
  UZU_CONST uint GROUP_LANES = GROUP_LANES_;
  UZU_CONST uint NUM_SIMDGROUPS = NUM_SIMDGROUPS_;
  UZU_CONST uint K_SPLIT = K_SPLIT_;
  UZU_CONST uint ROW_BLOCKS = METAL_SIMD_SIZE / REDUCTION_LANES;
  UZU_CONST uint SIMDGROUP_OUTPUT_ROWS = OUTPUT_ROWS / (NUM_SIMDGROUPS / K_SPLIT);
  UZU_CONST uint ROWS_PER_LANE = SIMDGROUP_OUTPUT_ROWS / ROW_BLOCKS;
  UZU_CONST uint GROUPS_PER_STEP = REDUCTION_LANES / GROUP_LANES;

  static_assert(INPUT_ROWS > 0 && OUTPUT_ROWS > 0, "GEMV tiles must be non-empty");
  static_assert(
      REDUCTION_LANES == 4 || REDUCTION_LANES == 8 || REDUCTION_LANES == 16 || REDUCTION_LANES == 32,
      "reduction lanes must divide one simdgroup"
  );
  static_assert(METAL_SIMD_SIZE % REDUCTION_LANES == 0, "reduction lanes must divide one simdgroup");
  static_assert(GROUP_LANES > 0 && REDUCTION_LANES % GROUP_LANES == 0, "group lanes must divide reduction lanes");
  static_assert(NUM_SIMDGROUPS > 0 && NUM_SIMDGROUPS % K_SPLIT == 0, "K split must divide simdgroups");
  static_assert(OUTPUT_ROWS % (NUM_SIMDGROUPS / K_SPLIT) == 0, "output rows must divide row groups");
  static_assert(SIMDGROUP_OUTPUT_ROWS % ROW_BLOCKS == 0, "output rows must divide reduction blocks");

  template <typename Fn>
  static METAL_FUNC void for_each_input_row(Fn fn) {
    const_for_loop<0, INPUT_ROWS, 1>(fn);
  }

  template <typename Fn>
  static METAL_FUNC void for_each_output_row(Fn fn) {
    const_for_loop<0, ROWS_PER_LANE, 1>(fn);
  }
};

template <typename Tile, bool FULL_TILE>
struct OutputTile {
  uint simd_group;
  uint simd_lane;
  uint reduction_lane;
  uint row_group;
  uint k_slice;
  uint input_row;
  uint tile_row;
  uint local_row;
  uint row0;
  bool writer;

  static METAL_FUNC OutputTile make(uint output_tile_idx, uint input_tile_idx, uint simd_group, uint simd_lane) {
    const uint row_group = simd_group / Tile::K_SPLIT;
    const uint k_slice = simd_group % Tile::K_SPLIT;
    const uint row_block = simd_lane / Tile::REDUCTION_LANES;
    const uint local_row = row_group * Tile::SIMDGROUP_OUTPUT_ROWS + row_block * Tile::ROWS_PER_LANE;
    OutputTile tile;
    tile.simd_group = simd_group;
    tile.simd_lane = simd_lane;
    tile.reduction_lane = simd_lane % Tile::REDUCTION_LANES;
    tile.row_group = row_group;
    tile.k_slice = k_slice;
    tile.input_row = input_tile_idx * Tile::INPUT_ROWS;
    tile.tile_row = output_tile_idx * Tile::OUTPUT_ROWS;
    tile.local_row = local_row;
    tile.row0 = tile.tile_row + local_row;
    tile.writer = Tile::K_SPLIT == 1 || k_slice == 0;
    return tile;
  }

  static METAL_FUNC bool row_in_range(uint row, uint out_vec_size) { return FULL_TILE || row < out_vec_size; }
};

} // namespace gemm
} // namespace uzu
