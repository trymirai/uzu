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

template <GemmBPrologueKind B_PROLOGUE, typename Loader, typename ElementType>
static METAL_FUNC auto make_loader(
    const device uint8_t* storage,
    const device ElementType* scales,
    const device ElementType* biases,
    const device uint8_t* zero_points,
    const bool signed_codes,
    const int k_elements,
    const int groups_per_row,
    threadgroup ElementType* shared,
    const ushort simdgroup_index,
    const ushort simd_lane_id
) {
  if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
    return Loader(storage, scales, biases, signed_codes, k_elements, shared, simdgroup_index, simd_lane_id);
  } else if constexpr (
      B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant || B_PROLOGUE == GemmBPrologueKind::ScaleSymmetricDequant
  ) {
    return Loader(
        storage,
        scales,
        zero_points,
        signed_codes,
        k_elements,
        groups_per_row,
        shared,
        simdgroup_index,
        simd_lane_id
    );
  } else {
    static_assert(B_PROLOGUE != B_PROLOGUE, "unsupported quantized loader prologue");
  }
}

} // namespace gemm
} // namespace uzu
