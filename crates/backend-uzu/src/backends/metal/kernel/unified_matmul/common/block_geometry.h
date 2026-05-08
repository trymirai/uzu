#pragma once

#include "../../matmul/common/defines.h"
#include "../../generated/matmul.h"

using namespace metal;

namespace uzu {
namespace unified_gemm {

static METAL_FUNC uint2 swizzled_block_id(
    uint2 threadgroup_position,
    uint swizzle_log
) {
  const uint stride = pow2(swizzle_log);
  return uint2(
      threadgroup_position.x / stride,
      threadgroup_position.y * stride + (threadgroup_position.x % stride));
}

template <uint BLOCK_M, uint BLOCK_N>
struct BlockGeometry {
  uint2 tile_id;
  uint block_row_start;
  uint block_col_start;
  bool out_of_bounds;

  static METAL_FUNC BlockGeometry compute(
      uint2 tile_id,
      const constant uzu::matmul::GemmParams* params
  ) {
    BlockGeometry geometry;
    geometry.tile_id = tile_id;
    geometry.out_of_bounds = (tile_id.x >= params->threadgroups_per_row)
                          || (tile_id.y >= params->threadgroups_per_column);
    geometry.block_row_start = tile_id.y * BLOCK_M;
    geometry.block_col_start = tile_id.x * BLOCK_N;
    return geometry;
  }
};

} // namespace unified_gemm
} // namespace uzu
