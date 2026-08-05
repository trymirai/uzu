#pragma once

#include <metal_stdlib>

#include "../../../../generated/gemm.h"
#include "../quant_scale_bias.h"
#include "../quant_scale_zero_point.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <
    typename T,
    short THREADGROUP_TILE_ROWS,
    short THREADGROUP_TILE_COLS,
    short DESTINATION_LEADING_DIMENSION,
    short REDUCTION_DIMENSION,
    short THREADGROUP_SIZE,
    short GROUP_SIZE,
    short BITS>
struct QuantizedLoaders {
  using ScaleBias = QuantizedBlockLoaderScaleBias<
      T,
      THREADGROUP_TILE_ROWS,
      THREADGROUP_TILE_COLS,
      DESTINATION_LEADING_DIMENSION,
      REDUCTION_DIMENSION,
      THREADGROUP_SIZE,
      GROUP_SIZE,
      BITS>;
  using ScaleZeroPoint = QuantizedBlockLoaderScaleZeroPoint<
      T,
      THREADGROUP_TILE_ROWS,
      THREADGROUP_TILE_COLS,
      DESTINATION_LEADING_DIMENSION,
      REDUCTION_DIMENSION,
      THREADGROUP_SIZE,
      GROUP_SIZE,
      BITS>;
  using ScaleSymmetric = QuantizedBlockLoaderScaleZeroPoint<
      T,
      THREADGROUP_TILE_ROWS,
      THREADGROUP_TILE_COLS,
      DESTINATION_LEADING_DIMENSION,
      REDUCTION_DIMENSION,
      THREADGROUP_SIZE,
      GROUP_SIZE,
      BITS,
      true>;
};

template <GemmBPrologueKind SCHEME, typename Loaders, typename RightArgs, typename ElementType>
static METAL_FUNC auto make_quantized_loader(
    const thread RightArgs& right,
    const int k_elements,
    const int groups_per_row,
    threadgroup ElementType* shared,
    const ushort simdgroup_index,
    const ushort simd_lane_id
) {
  if constexpr (SCHEME == GemmBPrologueKind::ScaleBiasDequant) {
    return typename Loaders::ScaleBias(
        right.storage.values,
        right.scales,
        right.biases,
        right.storage.signed_codes,
        k_elements,
        shared,
        simdgroup_index,
        simd_lane_id
    );
  } else if constexpr (SCHEME == GemmBPrologueKind::ScaleZeroPointDequant) {
    return typename Loaders::ScaleZeroPoint(
        right.storage.values,
        right.scales,
        right.zero_points,
        right.storage.signed_codes,
        k_elements,
        groups_per_row,
        shared,
        simdgroup_index,
        simd_lane_id
    );
  } else if constexpr (SCHEME == GemmBPrologueKind::ScaleSymmetricDequant) {
    return typename Loaders::ScaleSymmetric(
        right.storage.values,
        right.scales,
        right.storage.signed_codes,
        k_elements,
        groups_per_row,
        shared,
        simdgroup_index,
        simd_lane_id
    );
  } else {
    static_assert(
        SCHEME == GemmBPrologueKind::ScaleBiasDequant || SCHEME == GemmBPrologueKind::ScaleZeroPointDequant ||
            SCHEME == GemmBPrologueKind::ScaleSymmetricDequant,
        "unsupported quantized loader scheme"
    );
  }
}

} // namespace gemm
} // namespace uzu
