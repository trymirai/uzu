#pragma once

#include "../../../common/integral_constant.h"
#include "../../../common/thread_context.h"
#include "../../common/fragment.h"
#include "../../common/mxu_fragment_ops.h"
#include "../../common/mxu_gemm_loop.h"
#include "../../../generated/matmul.h"
#include "../generated/gemm.h"
#include "block_geometry.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <
    typename T,
    ushort BLOCK_M,
    ushort BLOCK_N,
    ushort BLOCK_K,
    ushort SIMDGROUPS_PER_ROW,
    ushort SIMDGROUPS_PER_COLUMN,
    bool VALID =
        (BLOCK_M % SIMDGROUPS_PER_ROW == 0 &&
         BLOCK_N % SIMDGROUPS_PER_COLUMN == 0 &&
         (BLOCK_M / SIMDGROUPS_PER_ROW) % 16 == 0 &&
         (BLOCK_N / SIMDGROUPS_PER_COLUMN) % 16 == 0)>
struct MxuMmaCore {
  METAL_CONST ushort SIMDGROUP_BLOCK_M = BLOCK_M / SIMDGROUPS_PER_ROW;
  METAL_CONST ushort SIMDGROUP_BLOCK_N = BLOCK_N / SIMDGROUPS_PER_COLUMN;
  METAL_CONST ushort SIMDGROUP_BLOCK_K = 32;
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
      GemmOutputTransformKind output_transform,
      const thread ThreadContext& thread_context
  ) {
    const uint simd_group_id = thread_context.simdgroup_index;
    const uint2 tile_id = block_id(thread_context.threadgroup_position.xy, params);
    const auto geometry =
        BlockGeometry<BLOCK_M, BLOCK_N>::compute(tile_id, params);
    if (geometry.out_of_bounds) {
      return;
    }

    const size_t block_row = size_t(geometry.block_row_start);
    const size_t block_col = size_t(geometry.block_col_start);

    const device T* activations_block =
        activations + block_row * params->leading_dimension_a;
    const device T* weights_block =
        weights + block_col * params->leading_dimension_b;

    const ushort tile_row_offset =
        SIMDGROUP_BLOCK_M * (simd_group_id / SIMDGROUPS_PER_COLUMN);
    const ushort tile_col_offset =
        SIMDGROUP_BLOCK_N * (simd_group_id % SIMDGROUPS_PER_COLUMN);

    device T* result_simdgroup =
        result + block_row * params->leading_dimension_d + block_col +
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

    const bool apply_scale =
        output_transform == GemmOutputTransformKind::Scale ||
        output_transform == GemmOutputTransformKind::ScaleAccumulate;
    const bool apply_accumulate =
        output_transform == GemmOutputTransformKind::Accumulate ||
        output_transform == GemmOutputTransformKind::ScaleAccumulate;

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

                  if (apply_scale) {
                    const AccumulatorType scale = AccumulatorType(params->ab_scale);
                    METAL_PRAGMA_UNROLL
                    for (ushort i = 0; i < accumulator_tile.ELEMENTS_PER_TILE;
                         i++) {
                      accumulator_tile.elements()[i] *= scale;
                    }
                  }

                  if (apply_accumulate) {
                    uzu::matmul::Fragment<T, TILES_M, TILES_N, uzu::matmul::MxuFragmentOps>
                        existing_output(thread_context);
                    if constexpr (aligned_m.value && aligned_n.value) {
                      existing_output.load(
                          result_simdgroup,
                          int(params->leading_dimension_d)
                      );
                    } else {
                      existing_output.load_safe(
                          result_simdgroup,
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

template <
    typename T,
    ushort BLOCK_M,
    ushort BLOCK_N,
    ushort BLOCK_K,
    ushort SIMDGROUPS_PER_ROW,
    ushort SIMDGROUPS_PER_COLUMN>
struct MxuMmaCore<
    T,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    SIMDGROUPS_PER_ROW,
    SIMDGROUPS_PER_COLUMN,
    false> {
  static METAL_FUNC void run(
      const device T*,
      const device T*,
      device T*,
      const constant uzu::matmul::GemmParams*,
      bool,
      bool,
      bool,
      GemmOutputTransformKind,
      const thread ThreadContext&
  ) {}
};

} // namespace gemm
} // namespace uzu
