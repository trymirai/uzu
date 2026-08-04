#pragma once

#include <metal_stdlib>

#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../common/fragment.h"
#include "../../../common/mxu_fragment/ops.h"
#include "../../../common/mxu_gemm_loop.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace schedules {

template <ushort BITS_VALUE, ushort RIGHT_GROUP_SIZE_VALUE>
struct StagedSchedule {
  using RightFormat = uzu::matmul::IntegerFormat<BITS_VALUE, uzu::matmul::Signedness::Signed>;
  UZU_CONST ushort RIGHT_GROUP_SIZE = RIGHT_GROUP_SIZE_VALUE;

  template <typename Core, bool ALIGNED_M, bool ALIGNED_N, typename Loader>
  static METAL_FUNC typename Core::AccumFragment run(
      const device typename Core::LeftType* a_simdgroup,
      threadgroup typename Core::RightType* b_shared,
      const int leading_dimension_a,
      const int aligned_k_iterations,
      const short simdgroup_limit_m,
      const short simdgroup_limit_n,
      const ushort tile_col_offset,
      const ushort tile_block_cols,
      thread Loader& loader_b,
      const thread ThreadContext& thread_context
  ) {
    using FragmentOps = typename Core::FragmentOps;
    typename Core::AccumFragment accumulator;
    accumulator.clear();

    threadgroup typename Core::RightType* b_shared_simdgroup = b_shared + tile_col_offset * Core::SHARED_STRIDE_B;
    const short2 tile_dimensions_b = short2(Core::QUANT_BK, tile_block_cols);

    METAL_PRAGMA_NO_UNROLL
    for (int outer_k = 0; outer_k < aligned_k_iterations; ++outer_k) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if constexpr (ALIGNED_N) {
        loader_b.load_unsafe();
      } else {
        loader_b.load_safe(tile_dimensions_b);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      METAL_PRAGMA_NO_UNROLL
      for (int inner_k = 0; inner_k < Core::QUANT_BK; inner_k += Core::SIMDGROUP_BLOCK_K) {
        uzu::matmul::Fragment<typename Core::LeftType, Core::TILES_M, Core::TILES_K, FragmentOps> left_tile;
        uzu::matmul::Fragment<typename Core::RightType, Core::TILES_N, Core::TILES_K, FragmentOps> right_tile;

        auto left_src = uzu::matmul::fragment_source(a_simdgroup + inner_k, leading_dimension_a);
        if constexpr (!ALIGNED_M) {
          left_src = left_src.bounded(simdgroup_limit_m, Core::SIMDGROUP_BLOCK_K);
        }
        left_tile.load_from(thread_context.simd_lane_id, left_src);

        right_tile.load_from(
            thread_context.simd_lane_id,
            uzu::matmul::fragment_source(b_shared_simdgroup + inner_k, int(Core::SHARED_STRIDE_B))
        );

        FragmentOps::template fragment_mma<false, true>(accumulator, left_tile, right_tile);
      }

      a_simdgroup += Core::QUANT_BK;
      loader_b.next();
    }

    return accumulator;
  }
};

} // namespace schedules
} // namespace gemm
} // namespace uzu
