#pragma once

#include "../../../common/integral_constant.h"
#include "../../../common/thread_context.h"
#include "../../common/fragment.h"
#include "../../common/mxu_fragment_ops.h"
#include "../../common/mxu_gemm_loop.h"
#include "../../../generated/matmul.h"
#include "../generated/gemm.h"
#include "block_geometry.h"
#include "gemm_tiling.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <typename T, GemmTiling GEMM_TILING, bool TRANSPOSE_B>
struct MxuMmaCore {
  METAL_CONST ushort THREADGROUP_BLOCK_M = gemm_tiling_block_m(GEMM_TILING);
  METAL_CONST ushort THREADGROUP_BLOCK_N = gemm_tiling_block_n(GEMM_TILING);
  METAL_CONST ushort SIMDGROUPS_PER_ROW =
      gemm_tiling_simdgroups_per_row(GEMM_TILING);
  METAL_CONST ushort SIMDGROUPS_PER_COLUMN =
      gemm_tiling_simdgroups_per_column(GEMM_TILING);
  METAL_CONST ushort SIMDGROUP_BLOCK_M =
      THREADGROUP_BLOCK_M / SIMDGROUPS_PER_ROW;
  METAL_CONST ushort SIMDGROUP_BLOCK_N =
      THREADGROUP_BLOCK_N / SIMDGROUPS_PER_COLUMN;
  METAL_CONST ushort SIMDGROUP_BLOCK_K = 32;
  METAL_CONST ushort THREADGROUP_BLOCK_K = 256;
  METAL_CONST ushort TILES_M =
      SIMDGROUP_BLOCK_M / uzu::matmul::MxuFragmentOps::FRAGMENT_ROWS;
  METAL_CONST ushort TILES_N =
      SIMDGROUP_BLOCK_N / uzu::matmul::MxuFragmentOps::FRAGMENT_COLS;

  using AccumulatorType = float;

  static METAL_FUNC void run(
      const device T* a,
      const device T* b,
      device T* d,
      const constant uzu::matmul::GemmParams* params,
      GemmAlignment alignment,
      GemmDTransform output_transform,
      const device T* output_bias,
      const thread ThreadContext& thread_context
  ) {
    (void)output_bias;
    const uint2 tile = tile_id(thread_context.threadgroup_position.xy, params);
    const auto geometry =
        ThreadgroupTileGeometry<THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N>::
            compute(tile, params);
    if (geometry.out_of_bounds) {
      return;
    }

    const size_t block_row = size_t(geometry.block_row_start);
    const size_t block_col = size_t(geometry.block_col_start);

    const device T* a_block = a + block_row * params->leading_dimension_a;
    const device T* b_block =
        b + (TRANSPOSE_B ? block_col * params->leading_dimension_b : block_col);

    const ushort tile_row_offset =
        SIMDGROUP_BLOCK_M *
        (thread_context.simdgroup_index / SIMDGROUPS_PER_COLUMN);
    const ushort tile_col_offset =
        SIMDGROUP_BLOCK_N *
        (thread_context.simdgroup_index % SIMDGROUPS_PER_COLUMN);

    device T* d_simdgroup =
        d + block_row * params->leading_dimension_d + block_col +
        tile_row_offset * params->leading_dimension_d + tile_col_offset;

    const short simdgroup_limit_m =
        alignment.contains(GemmAlignment::M)
            ? SIMDGROUP_BLOCK_M
            : short(
                  min(int(SIMDGROUP_BLOCK_M),
                      int(params->M) -
                          int(geometry.block_row_start + tile_row_offset))
              );
    const short simdgroup_limit_n =
        alignment.contains(GemmAlignment::N)
            ? SIMDGROUP_BLOCK_N
            : short(
                  min(int(SIMDGROUP_BLOCK_N),
                      int(params->N) -
                          int(geometry.block_col_start + tile_col_offset))
              );

    const device T* a_simdgroup =
        a_block + size_t(tile_row_offset) * params->leading_dimension_a;
    const device T* b_simdgroup =
        b_block + (TRANSPOSE_B ? size_t(tile_col_offset) *
                                     int(params->leading_dimension_b)
                               : size_t(tile_col_offset));

    const int aligned_k_iterations = int(params->K) / int(THREADGROUP_BLOCK_K);

    const bool apply_scale = output_transform.contains(GemmDTransform::SCALE);
    const bool apply_accumulate =
        output_transform.contains(GemmDTransform::ACCUMULATE);

    dispatch_bool(alignment.contains(GemmAlignment::K), [&](auto aligned_k) {
      dispatch_bool(
          alignment.contains(GemmAlignment::M) ||
              (simdgroup_limit_m == SIMDGROUP_BLOCK_M),
          [&](auto aligned_m) {
            dispatch_bool(
                alignment.contains(GemmAlignment::N) ||
                    (simdgroup_limit_n == SIMDGROUP_BLOCK_N),
                [&](auto aligned_n) {
                  auto accumulator_tile = uzu::matmul::gemm_loop<
                      T,
                      SIMDGROUP_BLOCK_M,
                      SIMDGROUP_BLOCK_N,
                      SIMDGROUP_BLOCK_K,
                      THREADGROUP_BLOCK_K,
                      false,
                      TRANSPOSE_B,
                      aligned_m.value,
                      aligned_n.value,
                      aligned_k.value,
                      AccumulatorType>(
                      a_simdgroup,
                      b_simdgroup,
                      int(params->leading_dimension_a),
                      int(params->leading_dimension_b),
                      int(params->K),
                      aligned_k_iterations,
                      simdgroup_limit_m,
                      simdgroup_limit_n,
                      thread_context
                  );

                  if (apply_scale) {
                    const AccumulatorType scale =
                        AccumulatorType(params->ab_scale);
                    METAL_PRAGMA_UNROLL
                    for (ushort i = 0; i < accumulator_tile.ELEMENTS_PER_TILE;
                         i++) {
                      accumulator_tile.elements()[i] *= scale;
                    }
                  }

                  if (apply_accumulate) {
                    uzu::matmul::Fragment<
                        T,
                        TILES_M,
                        TILES_N,
                        uzu::matmul::MxuFragmentOps>
                        existing_output(thread_context);
                    if constexpr (aligned_m.value && aligned_n.value) {
                      existing_output.load(
                          d_simdgroup,
                          int(params->leading_dimension_d)
                      );
                    } else {
                      existing_output.load_safe(
                          d_simdgroup,
                          int(params->leading_dimension_d),
                          short2(simdgroup_limit_n, simdgroup_limit_m)
                      );
                    }
                    METAL_PRAGMA_UNROLL
                    for (ushort i = 0; i < accumulator_tile.ELEMENTS_PER_TILE;
                         i++) {
                      accumulator_tile.elements()[i] +=
                          AccumulatorType(existing_output.elements()[i]);
                    }
                  }

                  if constexpr (aligned_m.value && aligned_n.value) {
                    accumulator_tile.store(
                        d_simdgroup,
                        int(params->leading_dimension_d)
                    );
                  } else {
                    accumulator_tile.store_safe(
                        d_simdgroup,
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

} // namespace gemm
} // namespace uzu
