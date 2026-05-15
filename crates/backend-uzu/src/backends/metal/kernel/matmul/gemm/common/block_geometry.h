#pragma once

#include "../../common/defines.h"
#include "../../../generated/matmul.h"

using namespace metal;

namespace uzu {
namespace gemm {

static METAL_FUNC uint morton_expand_bits(uint x) {
  x &= 0x55555555u;
  x = (x | (x >> 1)) & 0x33333333u;
  x = (x | (x >> 2)) & 0x0F0F0F0Fu;
  x = (x | (x >> 4)) & 0x00FF00FFu;
  x = (x | (x >> 8)) & 0x0000FFFFu;
  return x;
}

static METAL_FUNC uint2 morton_block_id(uint2 threadgroup_position) {
  return uint2(
      morton_expand_bits(threadgroup_position.x),
      morton_expand_bits(threadgroup_position.x >> 1)
  );
}

static METAL_FUNC uint2 block_id(
    uint2 threadgroup_position,
    const constant uzu::matmul::GemmParams* params
) {
  if (params->use_morton) {
    return morton_block_id(threadgroup_position);
  }
  return threadgroup_position;
}

// Per-threadgroup placement in the output matrix `D`: the (possibly
// morton/swizzle-rewritten) tile id, whether that tile is in-bounds for the
// actual grid (morton padding can produce out-of-range ids), and the row/col
// origin of this threadgroup's tile in `D`.
template <uint THREADGROUP_BLOCK_M, uint THREADGROUP_BLOCK_N>
struct ThreadgroupTileGeometry {
  uint2 tile_id;
  uint block_row_start;
  uint block_col_start;
  bool out_of_bounds;

  static METAL_FUNC ThreadgroupTileGeometry
  compute(uint2 tile_id, const constant uzu::matmul::GemmParams* params) {
    ThreadgroupTileGeometry geometry;
    geometry.tile_id = tile_id;
    geometry.out_of_bounds = (tile_id.x >= params->threadgroups_per_row) ||
                             (tile_id.y >= params->threadgroups_per_column);
    geometry.block_row_start = tile_id.y * THREADGROUP_BLOCK_M;
    geometry.block_col_start = tile_id.x * THREADGROUP_BLOCK_N;
    return geometry;
  }
};

} // namespace gemm
} // namespace uzu
