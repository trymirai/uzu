#pragma once

#include "../../../../common/integral_constant.h"
#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../common/fragment.h"
#include "../../../common/mxu_gemm_loop.h"
#include "../gemm_alignment.h"
#include "tile_context.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace schedules {

struct DenseSchedule {
  template <typename Core, bool ALIGNED_M, bool ALIGNED_N>
  static METAL_FUNC typename Core::AccumFragment launch(
      typename Core::LeftArgs left,
      typename Core::RightArgs right,
      threadgroup typename Core::RightElementType*,
      const constant uzu::matmul::GemmParams* params,
      const TileContext tile,
      const GemmAlignment alignment,
      const thread ThreadContext& thread_context
  ) {
    right.template seek_columns<Core::TRANSPOSE_RIGHT>(
        tile.block_col + tile.tile_col_offset,
        tile.k_offset,
        params->leading_dimension_b
    );

    typename Core::AccumFragment accumulator;
    dispatch_bool(alignment.contains(GemmAlignment::K), [&](auto aligned_k) {
      accumulator = uzu::matmul::gemm_loop<
          typename Core::LeftElementType,
          typename Core::RightElementType,
          Core::SIMDGROUP_BLOCK_M,
          Core::SIMDGROUP_BLOCK_N,
          Core::SIMDGROUP_BLOCK_K,
          Core::THREADGROUP_BLOCK_K_FP,
          false,
          Core::TRANSPOSE_RIGHT,
          ALIGNED_M,
          ALIGNED_N,
          aligned_k.value,
          typename Core::AccumulatorType>(
          left.values,
          right.values,
          int(params->leading_dimension_a),
          int(params->leading_dimension_b),
          int(params->K),
          int(params->aligned_inner_iterations),
          tile.simdgroup_limit_m,
          tile.simdgroup_limit_n,
          thread_context
      );
    });
    return accumulator;
  }
};

} // namespace schedules
} // namespace gemm
} // namespace uzu
