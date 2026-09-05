#pragma once

#include <metal_stdlib>

#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../common/fragment.h"
#include "../../../common/mxu_gemm_loop.h"
#include "../gemm_alignment.h"
#include "../microfloat_loader.h"
#include "../operands.h"
#include "../quant_scale_bias.h"
#include "../quant_scale_zero_point.h"
#include "tile_context.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace schedules {

template <typename Core, typename RightOperand>
static METAL_FUNC auto make_mxfp4_loader(
    const typename Core::RightStorage right,
    const constant uzu::matmul::GemmParams* params,
    const size_t block_col,
    const uint k_offset,
    threadgroup typename Core::RightElementType* staging,
    const thread ThreadContext& thread_context
) {
  static_assert(RightOperand::MICROFLOAT, "MXFP4 loader requires MXFP4 weights");
  using Element = typename Core::RightElementType;
  const size_t code_row_stride = size_t(params->K) / 2;
  const size_t scale_row_stride = size_t(params->K) / RightOperand::GROUP_SIZE;
  const device uint8_t* codes = right.codes + block_col * code_row_stride + size_t(k_offset) / 2;
  const device uint8_t* scales = right.microfloat_scales + block_col * scale_row_stride;
  using Loader = Mxfp4BlockLoader<
      Element,
      Core::THREADGROUP_BLOCK_N,
      Core::THREADGROUP_BLOCK_K,
      Core::SHARED_STRIDE_B,
      Core::THREADGROUP_THREADS,
      RightOperand::GROUP_SIZE>;
  return Loader(
      codes,
      scales,
      code_row_stride,
      scale_row_stride,
      k_offset,
      staging,
      thread_context.simdgroup_index,
      thread_context.simd_lane_id
  );
}

template <typename Core, typename RightOperand>
static METAL_FUNC auto make_integer_loader(
    const typename Core::RightStorage right,
    const constant uzu::matmul::GemmParams* params,
    const size_t block_col,
    const uint k_offset,
    threadgroup typename Core::RightElementType* staging,
    const thread ThreadContext& thread_context
) {
  using Element = typename Core::RightElementType;
  using Format = typename RightOperand::Format;
  const uint row_stride =
      uint(params->K) * uint(get_bytes_per_pack<RightOperand::BITS>()) / uint(get_pack_factor<RightOperand::BITS>());
  const uint groups_per_row = (uint(params->K) + uint(RightOperand::GROUP_SIZE) - 1) / uint(RightOperand::GROUP_SIZE);
  const uint first_group = k_offset / uint(RightOperand::GROUP_SIZE);
  const device uint8_t* values = right.codes + size_t(block_col) * row_stride +
                                 size_t(k_offset) * size_t(get_bytes_per_pack<RightOperand::BITS>()) /
                                     size_t(get_pack_factor<RightOperand::BITS>());
  const device Element* scales = right.scales + block_col * groups_per_row + first_group;

  if constexpr (RightOperand::SCHEME == GemmBPrologueKind::ScaleBiasDequant) {
    using Loader = QuantizedBlockLoaderScaleBias<
        Element,
        Core::THREADGROUP_BLOCK_N,
        Core::THREADGROUP_BLOCK_K,
        Core::SHARED_STRIDE_B,
        1,
        Core::THREADGROUP_THREADS,
        RightOperand::GROUP_SIZE,
        Format::BITS>;
    return Loader(
        values,
        scales,
        right.bias() + block_col * groups_per_row + first_group,
        right.signed_codes,
        int(params->K),
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
        Format::BITS>;
    const device uint8_t* zero_points = right.zp() + block_col * zero_point_row_stride<Format::BITS>(groups_per_row) +
                                        ((Format::BITS == 4) ? first_group / 2 : first_group);
    return Loader(
        values,
        scales,
        zero_points,
        right.signed_codes,
        int(params->K),
        int(groups_per_row),
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
        Format::BITS,
        true>;
    return Loader(
        values,
        scales,
        right.signed_codes,
        int(params->K),
        int(groups_per_row),
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

template <typename Core>
static METAL_FUNC auto make_staged_loader(
    const typename Core::RightStorage right,
    const constant uzu::matmul::GemmParams* params,
    const size_t block_col,
    const uint k_offset,
    threadgroup typename Core::RightElementType* staging,
    const thread ThreadContext& thread_context
) {
  using Right = typename Core::Right;
  if constexpr (Right::DENSE) {
    return make_full_precision_loader<Core>(right, params, block_col, k_offset, staging, thread_context);
  } else if constexpr (Right::MICROFLOAT) {
    return make_mxfp4_loader<Core, Right>(right, params, block_col, k_offset, staging, thread_context);
  } else {
    return make_integer_loader<Core, Right>(right, params, block_col, k_offset, staging, thread_context);
  }
}

struct StagedSchedule {
  template <typename Core, bool ALIGNED_M, bool ALIGNED_N, bool, bool>
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
    auto loader = make_staged_loader<Core>(right, params, tile.block_col, tile.k_offset, staging, thread_context);

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
