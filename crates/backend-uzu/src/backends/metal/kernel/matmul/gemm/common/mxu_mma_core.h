#pragma once

#include "../../../common/integral_constant.h"
#include "../../../common/thread_context.h"
#include "../../../hadamard_transform/hadamard_transform.h"
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
      const device int32_t* rht_factors,
      const thread ThreadContext& thread_context
  ) {
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
    const bool apply_bias = output_transform.contains(GemmDTransform::BIAS);

    const device T* bias_simdgroup = output_bias + size_t(block_col) +
        size_t(tile_col_offset);
    using FragType =
        uzu::matmul::Fragment<T, TILES_M, TILES_N, uzu::matmul::MxuFragmentOps>;
    const short2 thread_position = FragType::get_position(thread_context);

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

                  if (apply_bias) {
                    METAL_PRAGMA_UNROLL
                    for (ushort tile_row = 0; tile_row < TILES_M; ++tile_row) {
                      METAL_PRAGMA_UNROLL
                      for (ushort tile_col = 0; tile_col < TILES_N;
                           ++tile_col) {
                        thread auto& frag =
                            accumulator_tile.fragment_at(tile_row, tile_col);
                        const short col_base = short(
                            tile_col *
                                uzu::matmul::MxuFragmentOps::FRAGMENT_COLS +
                            thread_position.x
                        );
                        METAL_PRAGMA_UNROLL
                        for (ushort i = 0;
                             i < uzu::matmul::MxuFragmentOps::
                                     THREAD_ELEMENT_ROWS;
                             ++i) {
                          METAL_PRAGMA_UNROLL
                          for (ushort j = 0;
                               j < uzu::matmul::MxuFragmentOps::
                                       THREAD_ELEMENT_COLS;
                               ++j) {
                            const ushort element_index = i *
                                    uzu::matmul::MxuFragmentOps::
                                        THREAD_ELEMENT_COLS +
                                j;
                            const short col_local = short(col_base + j);
                            if constexpr (aligned_n.value) {
                              frag[element_index] +=
                                  AccumulatorType(bias_simdgroup[col_local]);
                            } else {
                              if (col_local < simdgroup_limit_n) {
                                frag[element_index] +=
                                    AccumulatorType(bias_simdgroup[col_local]);
                              }
                            }
                          }
                        }
                      }
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

    if (output_transform.contains(GemmDTransform::RHT)) {
      threadgroup_barrier(mem_flags::mem_device);
      device T* d_block = d + block_row * params->leading_dimension_d +
          block_col;
      const device int32_t* rht_factors_block = rht_factors + block_col;
      const ushort tile_block_rows =
          ushort(min(int(THREADGROUP_BLOCK_M),
                     int(params->M) - int(block_row)));
      const ushort tile_block_cols =
          ushort(min(int(THREADGROUP_BLOCK_N),
                     int(params->N) - int(block_col)));
      constexpr ushort SIMDGROUP_COUNT =
          SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN;
      const ushort stripes_per_row = tile_block_cols / METAL_SIMD_SIZE;
      const ushort sg_index = thread_context.simdgroup_index;
      const ushort simd_lane = thread_context.simd_lane_id;
      const uint total_work = uint(tile_block_rows) * uint(stripes_per_row);
      for (uint w = sg_index; w < total_work; w += SIMDGROUP_COUNT) {
        const ushort row_local = ushort(w / stripes_per_row);
        const ushort stripe = ushort(w % stripes_per_row);
        const ushort col_local = stripe * ushort(METAL_SIMD_SIZE) + simd_lane;
        const size_t d_idx =
            size_t(row_local) * size_t(params->leading_dimension_d) +
            size_t(col_local);
        T value = d_block[d_idx];
        d_block[d_idx] = simdgroup_output_random_hadamard_transform(
            simd_lane, value, rht_factors_block[col_local]
        );
      }
    }
  }
};

} // namespace gemm
} // namespace uzu
