#pragma once

#include <metal_stdlib>

#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../common/fragment.h"
#include "../../../common/mxu_gemm_loop.h"
#include "../gemm_alignment.h"
#include "../operands.h"
#include "../quant_scale_bias.h"
#include "../quant_scale_zero_point.h"
#include "tile_context.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace schedules {

template <typename Core, typename RightOperand>
static METAL_FUNC auto make_staged_loader(
    const typename Core::RightStorage right,
    const constant uzu::matmul::GemmParams* params,
    const size_t block_col,
    const uint k_offset,
    threadgroup typename Core::RightElementType* staging,
    const thread ThreadContext& thread_context
) {
  using Element = typename Core::RightElementType;
  const uint row_stride =
      uint(params->K) * uint(get_bytes_per_pack<RightOperand::BITS>()) / uint(get_pack_factor<RightOperand::BITS>());
  const uint groups_per_row = (uint(params->K) + uint(RightOperand::GROUP_SIZE) - 1) / uint(RightOperand::GROUP_SIZE);
  const uint first_group = k_offset / uint(RightOperand::GROUP_SIZE);
  const int params_group_stride = int(params->metadata_group_stride);
  const int params_output_stride = params_group_stride == 1 ? int(groups_per_row) : 1;
  const int params_offset = int(block_col) * params_output_stride + int(first_group) * params_group_stride;
  const device Element* scales = right.scales + params_offset;
  const device uint8_t* values = right.codes + size_t(block_col) * row_stride +
                                 size_t(k_offset) * size_t(get_bytes_per_pack<RightOperand::BITS>()) /
                                     size_t(get_pack_factor<RightOperand::BITS>());
  if constexpr (RightOperand::SCHEME == GemmBPrologueKind::ScaleBiasDequant) {
    using Loader = QuantizedBlockLoaderScaleBias<
        Element,
        Core::THREADGROUP_BLOCK_N,
        Core::THREADGROUP_BLOCK_K,
        Core::SHARED_STRIDE_B,
        1,
        Core::THREADGROUP_THREADS,
        RightOperand::GROUP_SIZE,
        RightOperand::BITS>;
    return Loader(
        values,
        scales,
        right.bias() + params_offset,
        right.signed_codes,
        int(params->K),
        params_group_stride,
        staging,
        thread_context.simdgroup_index,
        thread_context.simd_lane_id
    );
  } else if constexpr (RightOperand::SCHEME == GemmBPrologueKind::ScaleZeroPointDequant) {
    using Loader = QuantizedBlockLoaderScaleZeroPoint<
        Element,
        Core::THREADGROUP_BLOCK_N,
        Core::THREADGROUP_BLOCK_K,
        Core::SHARED_STRIDE_B,
        1,
        Core::THREADGROUP_THREADS,
        RightOperand::GROUP_SIZE,
        RightOperand::BITS,
        false>;
    return Loader(
        values,
        scales,
        right.zp(),
        right.signed_codes,
        int(params->K),
        params_group_stride,
        uint(block_col) * zero_point_bit_stride<RightOperand::BITS>(params_output_stride) +
            first_group * zero_point_bit_stride<RightOperand::BITS>(params_group_stride),
        staging,
        thread_context.simdgroup_index,
        thread_context.simd_lane_id
    );
  } else {
    static_assert(
        RightOperand::SCHEME == GemmBPrologueKind::ScaleSymmetricDequant,
        "staged loader requires a quantized weight scheme"
    );
    using Loader = QuantizedBlockLoaderScaleZeroPoint<
        Element,
        Core::THREADGROUP_BLOCK_N,
        Core::THREADGROUP_BLOCK_K,
        Core::SHARED_STRIDE_B,
        1,
        Core::THREADGROUP_THREADS,
        RightOperand::GROUP_SIZE,
        RightOperand::BITS,
        true>;
    return Loader(
        values,
        scales,
        right.signed_codes,
        int(params->K),
        params_group_stride,
        staging,
        thread_context.simdgroup_index,
        thread_context.simd_lane_id
    );
  }
}

template <bool ALIGNED_N, typename Loader>
static METAL_FUNC void stage(thread Loader& loader, const short tile_block_cols, const short block_k) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if constexpr (ALIGNED_N) {
    loader.load_unsafe();
  } else {
    loader.load_safe(short2(block_k, tile_block_cols));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <typename Core>
static METAL_FUNC auto make_full_precision_loader(
    const typename Core::RightStorage right,
    const constant uzu::matmul::GemmParams* params,
    const size_t block_col,
    const uint k_offset,
    threadgroup typename Core::RightElementType* staging,
    const thread ThreadContext& thread_context
) {
  const device typename Core::RightElementType* values =
      right.dense + (Core::TRANSPOSE_RIGHT ? block_col * size_t(params->leading_dimension_b) + k_offset
                                           : block_col + size_t(k_offset) * size_t(params->leading_dimension_b));
  using Loader = typename Core::FullPrecisionRightLoader;
  return Loader(values, params->leading_dimension_b, staging, thread_context);
}

struct StagedSchedule {
  template <typename Core, bool ALIGNED_M, bool ALIGNED_N>
  static METAL_FUNC typename Core::AccumFragment launch(
      typename Core::LeftStorage left,
      typename Core::RightStorage right,
      threadgroup typename Core::RightElementType* staging,
      const constant uzu::matmul::GemmParams* params,
      const TileContext tile,
      const GemmAlignment,
      const thread ThreadContext& thread_context
  ) {
    static_assert(!Core::Left::QUANTIZED && Core::Right::QUANTIZED, "staged schedule requires dense A and quantized W");
    using LeftTile =
        uzu::matmul::Fragment<typename Core::LeftElementType, Core::TILES_M, Core::TILES_K, typename Core::FragmentOps>;
    using RightTile = uzu::matmul::Fragment<
        typename Core::RightElementType,
        Core::TILES_N,
        Core::TILES_K,
        typename Core::FragmentOps,
        uzu::matmul::ReadDirect,
        true>;

    auto left_source = uzu::matmul::fragment_source(
        left.values + size_t(tile.abs_row_base) * size_t(params->leading_dimension_a) + tile.k_offset,
        int(params->leading_dimension_a)
    );
    if constexpr (!ALIGNED_M) {
      left_source = left_source.bounded(tile.simdgroup_limit_m, Core::SIMDGROUP_BLOCK_K);
    }
    auto right_source = uzu::matmul::fragment_source(
        staging + tile.tile_col_offset * Core::SHARED_STRIDE_B,
        int(Core::SHARED_STRIDE_B)
    );
    auto loader = make_staged_loader<Core, typename Core::RightOperand>(
        right,
        params,
        tile.block_col,
        tile.k_offset,
        staging,
        thread_context
    );

    typename Core::AccumFragment accumulator;
    accumulator.clear();

    METAL_PRAGMA_NO_UNROLL
    for (int outer_k = 0; outer_k < int(params->aligned_inner_iterations); ++outer_k) {
      stage<ALIGNED_N>(loader, tile.tile_block_cols, Core::THREADGROUP_BLOCK_K);

      METAL_PRAGMA_NO_UNROLL
      for (ushort inner_k = 0; inner_k < Core::THREADGROUP_BLOCK_K; inner_k += Core::SIMDGROUP_BLOCK_K) {
        LeftTile left_tile;
        RightTile right_tile;
        left_tile.load_from(thread_context.simd_lane_id, left_source.advanced(inner_k));
        right_tile.load_from(thread_context.simd_lane_id, right_source.advanced(inner_k));
        uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
      }

      left_source = left_source.advanced(Core::THREADGROUP_BLOCK_K);
      loader.next();
    }

    return accumulator;
  }
};

} // namespace schedules
} // namespace gemm
} // namespace uzu
