#pragma once

#include "../../common/fragment.h"
#include "../../common/mxu_fragment/ops.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <typename Element, typename FragmentOps, bool ALIGNED_M, ushort TILES_M, ushort TILES_K>
static METAL_FUNC uzu::matmul::Fragment<Element, TILES_M, TILES_K, FragmentOps> load_integer_tile(
    const device Element* activation,
    const int leading_dimension,
    const short simdgroup_limit_m,
    const ushort simd_lane_id
) {
  uzu::matmul::Fragment<Element, TILES_M, TILES_K, FragmentOps> tile;
  auto source = uzu::matmul::fragment_source(activation, leading_dimension);
  if constexpr (!ALIGNED_M) {
    source = source.bounded(simdgroup_limit_m, TILES_K * FragmentOps::FRAGMENT_ROWS);
  }
  tile.load_from(simd_lane_id, source);
  return tile;
}

} // namespace gemm
} // namespace uzu
