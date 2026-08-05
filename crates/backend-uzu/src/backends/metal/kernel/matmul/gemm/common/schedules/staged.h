#pragma once

#include <metal_stdlib>

#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../common/fragment.h"
#include "../../../common/mxu_fragment/ops.h"
#include "../../../common/mxu_gemm_loop.h"
#include "../gemm_alignment.h"
#include "../integer_source.h"
#include "../operands.h"
#include "../quantized/loader.h"
#include "../quantized/slice.h"
#include "tile_context.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace schedules {

template <typename RightOperand, ushort DENSE_BLOCK_K, ushort SIMDGROUP_BLOCK_K>
struct OuterBlockK;

template <typename ElementType, ushort DENSE_BLOCK_K, ushort SIMDGROUP_BLOCK_K>
struct OuterBlockK<operands::Dense<ElementType>, DENSE_BLOCK_K, SIMDGROUP_BLOCK_K> {
  UZU_CONST ushort VALUE = DENSE_BLOCK_K;
};

template <
    typename Format,
    ushort GROUP_SIZE,
    GemmBPrologueKind SCHEME,
    typename ElementType,
    ushort DENSE_BLOCK_K,
    ushort SIMDGROUP_BLOCK_K>
struct OuterBlockK<operands::Quantized<Format, GROUP_SIZE, SCHEME, ElementType>, DENSE_BLOCK_K, SIMDGROUP_BLOCK_K> {
  UZU_CONST ushort VALUE = GROUP_SIZE;

  static_assert(VALUE % SIMDGROUP_BLOCK_K == 0, "quantized group size must contain complete MMA chunks");
  static_assert(DENSE_BLOCK_K % VALUE == 0, "tile block K must contain complete quantized groups");
};

template <typename RightOperand>
struct StagedRightLoader;

template <typename ElementType>
struct StagedRightLoader<operands::Dense<ElementType>> {
  template <typename Core>
  static METAL_FUNC typename Core::FullPrecisionRightLoader make(
      typename Core::RightArgs right,
      const constant uzu::matmul::GemmParams* params,
      const size_t block_col,
      const uint k_offset,
      threadgroup typename Core::RightElementType* staging,
      const thread ThreadContext& thread_context
  ) {
    right.template seek_columns<Core::TRANSPOSE_RIGHT>(block_col, k_offset, params->leading_dimension_b);
    return typename Core::FullPrecisionRightLoader(right.values, params->leading_dimension_b, staging, thread_context);
  }
};

template <typename Format, ushort GROUP_SIZE, GemmBPrologueKind SCHEME, typename ElementType>
struct StagedRightLoader<operands::Quantized<Format, GROUP_SIZE, SCHEME, ElementType>> {
  template <typename Core>
  static METAL_FUNC auto make(
      typename Core::RightArgs right,
      const constant uzu::matmul::GemmParams* params,
      const size_t block_col,
      const uint k_offset,
      threadgroup typename Core::RightElementType* staging,
      const thread ThreadContext& thread_context
  ) {
    const int k_elements = int(params->K);
    const auto slice =
        make_quantized_slice<operands::Quantized<Format, GROUP_SIZE, SCHEME, ElementType>>(k_offset, k_elements);
    right.storage.seek_block(block_col, k_offset, slice.row_stride_bytes);
    operands::seek_quantized_metadata(right, block_col, slice.groups_per_row, slice.first_group);

    using Loaders = QuantizedLoaders<
        typename Core::RightElementType,
        Core::THREADGROUP_BLOCK_N,
        Core::THREADGROUP_BLOCK_K,
        Core::SHARED_STRIDE_B,
        1,
        Core::THREADGROUP_THREADS,
        GROUP_SIZE,
        Format::BITS>;

    return make_quantized_loader<SCHEME, Loaders>(
        right,
        k_elements,
        slice.groups_per_row,
        staging,
        thread_context.simdgroup_index,
        thread_context.simd_lane_id
    );
  }
};

// Bit width and group size come from the right operand descriptor.
struct StagedSchedule {
  template <typename Core, bool ALIGNED_M, bool ALIGNED_N>
  static METAL_FUNC typename Core::AccumFragment launch(
      typename Core::LeftArgs left,
      typename Core::RightArgs right,
      threadgroup typename Core::RightElementType* staging,
      const constant uzu::matmul::GemmParams* params,
      const TileContext tile,
      const GemmAlignment,
      const thread ThreadContext& thread_context
  ) {
    using FragmentOps = typename Core::FragmentOps;
    using RightOperand = typename Core::RightOperand;

    auto loader_b = StagedRightLoader<RightOperand>::template make<Core>(
        right,
        params,
        tile.block_col,
        tile.k_offset,
        staging,
        thread_context
    );

    typename Core::AccumFragment accumulator;
    accumulator.clear();

    threadgroup typename Core::RightElementType* staging_simdgroup =
        staging + tile.tile_col_offset * Core::SHARED_STRIDE_B;
    const short2 tile_dimensions_b = short2(Core::THREADGROUP_BLOCK_K, tile.tile_block_cols);

    METAL_PRAGMA_NO_UNROLL
    for (int outer_k = 0; outer_k < int(params->aligned_inner_iterations); ++outer_k) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if constexpr (ALIGNED_N) {
        loader_b.load_unsafe();
      } else {
        loader_b.load_safe(tile_dimensions_b);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      METAL_PRAGMA_NO_UNROLL
      for (int inner_k = 0; inner_k < Core::THREADGROUP_BLOCK_K; inner_k += Core::SIMDGROUP_BLOCK_K) {
        uzu::matmul::Fragment<typename Core::LeftElementType, Core::TILES_M, Core::TILES_K, FragmentOps> left_tile;
        uzu::matmul::Fragment<typename Core::RightElementType, Core::TILES_N, Core::TILES_K, FragmentOps> right_tile;

        auto left_src = uzu::matmul::fragment_source(left.values + inner_k, int(params->leading_dimension_a));
        if constexpr (!ALIGNED_M) {
          left_src = left_src.bounded(tile.simdgroup_limit_m, Core::SIMDGROUP_BLOCK_K);
        }
        left_tile.load_from(thread_context.simd_lane_id, left_src);

        right_tile.load_from(
            thread_context.simd_lane_id,
            uzu::matmul::fragment_source(staging_simdgroup + inner_k, int(Core::SHARED_STRIDE_B))
        );

        FragmentOps::template fragment_mma<false, true>(accumulator, left_tile, right_tile);
      }

      left.advance_k(Core::THREADGROUP_BLOCK_K);
      loader_b.next();
    }

    return accumulator;
  }
};

} // namespace schedules
} // namespace gemm
} // namespace uzu
