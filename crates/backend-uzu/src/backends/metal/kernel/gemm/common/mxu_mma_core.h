#pragma once

#include "../../common/integral_constant.h"
#include "../../common/thread_context.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/mxu_fragment_ops.h"
#include "../../matmul/common/mxu_gemm_loop.h"
#include "../../generated/matmul.h"
#include "block_geometry.h"

using namespace metal;

namespace uzu {
namespace unified_gemm {

template <
    typename T,
    ushort BLOCK_M,
    ushort BLOCK_N,
    ushort SIMDGROUPS_PER_ROW,
    ushort SIMDGROUPS_PER_COLUMN>
struct MxuMmaCore {
  METAL_CONST ushort SIMDGROUP_BLOCK_M = BLOCK_M / SIMDGROUPS_PER_ROW;
  METAL_CONST ushort SIMDGROUP_BLOCK_N = BLOCK_N / SIMDGROUPS_PER_COLUMN;
  METAL_CONST ushort SIMDGROUP_BLOCK_K = 32;
  METAL_CONST ushort BLOCK_K = 256;
  METAL_CONST ushort TILES_M =
      SIMDGROUP_BLOCK_M / uzu::matmul::MxuFragmentOps::FRAGMENT_ROWS;
  METAL_CONST ushort TILES_N =
      SIMDGROUP_BLOCK_N / uzu::matmul::MxuFragmentOps::FRAGMENT_COLS;

  using AccumulatorType = float;

  static METAL_FUNC void run(
      const device T* activations,
      const device T* weights,
      device T* result,
      const constant uzu::matmul::GemmParams* params,
      const bool align_m,
      const bool align_n,
      const bool align_k,
      uint simd_group_id,
      uint2 threadgroup_position,
      const thread ThreadContext& thread_context
  ) {
    const uint2 tile_id =
        morton_block_id(threadgroup_position, params->use_morton);
    const auto geometry =
        BlockGeometry<BLOCK_M, BLOCK_N>::compute(tile_id, params);
    if (geometry.out_of_bounds) {
      return;
    }

    const size_t block_row_long = size_t(geometry.block_row_start);
    const size_t block_col_long = size_t(geometry.block_col_start);

    const device T* activations_block =
        activations + block_row_long * params->leading_dimension_a;
    const device T* weights_block =
        weights + block_col_long * params->leading_dimension_b;

    const ushort tile_row_offset =
        SIMDGROUP_BLOCK_M * (simd_group_id / SIMDGROUPS_PER_COLUMN);
    const ushort tile_col_offset =
        SIMDGROUP_BLOCK_N * (simd_group_id % SIMDGROUPS_PER_COLUMN);

    device T* result_simdgroup =
        result + block_row_long * params->leading_dimension_d + block_col_long +
        tile_row_offset * params->leading_dimension_d + tile_col_offset;

    const short simdgroup_limit_m =
        align_m ? SIMDGROUP_BLOCK_M
                : short(
                      min(int(SIMDGROUP_BLOCK_M),
                          int(params->M) -
                              int(geometry.block_row_start + tile_row_offset))
                  );
    const short simdgroup_limit_n =
        align_n ? SIMDGROUP_BLOCK_N
                : short(
                      min(int(SIMDGROUP_BLOCK_N),
                          int(params->N) -
                              int(geometry.block_col_start + tile_col_offset))
                  );

    const device T* activations_simdgroup =
        activations_block +
        size_t(tile_row_offset) * params->leading_dimension_a;
    const device T* weights_simdgroup =
        weights_block +
        size_t(tile_col_offset) * int(params->leading_dimension_b);

    const int aligned_k_iterations = int(params->K) / int(BLOCK_K);

    dispatch_bool(align_k, [&](auto aligned_k) {
      dispatch_bool(
          align_m || (simdgroup_limit_m == SIMDGROUP_BLOCK_M),
          [&](auto aligned_m) {
            dispatch_bool(
                align_n || (simdgroup_limit_n == SIMDGROUP_BLOCK_N),
                [&](auto aligned_n) {
                  auto accumulator_tile = uzu::matmul::gemm_loop<
                      T,
                      SIMDGROUP_BLOCK_M,
                      SIMDGROUP_BLOCK_N,
                      SIMDGROUP_BLOCK_K,
                      BLOCK_K,
                      false,
                      true,
                      aligned_m.value,
                      aligned_n.value,
                      aligned_k.value,
                      AccumulatorType>(
                      activations_simdgroup,
                      weights_simdgroup,
                      int(params->leading_dimension_a),
                      int(params->leading_dimension_b),
                      int(params->K),
                      aligned_k_iterations,
                      simdgroup_limit_m,
                      simdgroup_limit_n,
                      thread_context
                  );

                  if constexpr (aligned_m.value && aligned_n.value) {
                    accumulator_tile.store(
                        result_simdgroup,
                        int(params->leading_dimension_d)
                    );
                  } else {
                    accumulator_tile.store_safe(
                        result_simdgroup,
                        int(params->leading_dimension_d),
                        short2(simdgroup_limit_n, simdgroup_limit_m)
                    );
                  }
                }
            );
          }
      );
    });
  }
};

} // namespace unified_gemm
} // namespace uzu
