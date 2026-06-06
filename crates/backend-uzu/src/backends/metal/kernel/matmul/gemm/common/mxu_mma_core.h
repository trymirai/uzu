#pragma once

#include "../../../common/integral_constant.h"
#include "../../../common/thread_context.h"
#include "../../../hadamard_transform/hadamard_transform.h"
#include "../../common/fragment.h"
#include "gemm_rht.h"
#include "../../common/mxu_fragment_ops.h"
#include "../../common/mxu_gemm_loop.h"
#include "../../../generated/matmul.h"
#include "../generated/gemm.h"
#include "block_geometry.h"
#include "gemm_tiling.h"
#include "quant_pack.h"
#include "quant_scale_bias.h"
#include "quant_scale_zero_point.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <
    typename AT,
    typename BT,
    typename DT,
    GemmTiling GEMM_TILING,
    bool TRANSPOSE_B,
    GemmBPrologueKind B_PROLOGUE = GemmBPrologueKind::FullPrecision,
    int BITS = 0,
    int GROUP_SIZE = 0>
struct MxuMmaCore {
  METAL_CONST ushort THREADGROUP_BLOCK_M = gemm_tiling_block_m(GEMM_TILING);
  METAL_CONST ushort THREADGROUP_BLOCK_N = gemm_tiling_block_n(GEMM_TILING);
  METAL_CONST ushort SIMDGROUPS_PER_ROW = gemm_tiling_simdgroups_per_row(GEMM_TILING);
  METAL_CONST ushort SIMDGROUPS_PER_COLUMN = gemm_tiling_simdgroups_per_column(GEMM_TILING);
  METAL_CONST ushort SIMDGROUP_BLOCK_M = THREADGROUP_BLOCK_M / SIMDGROUPS_PER_ROW;
  METAL_CONST ushort SIMDGROUP_BLOCK_N = THREADGROUP_BLOCK_N / SIMDGROUPS_PER_COLUMN;
  METAL_CONST ushort SIMDGROUP_BLOCK_K = 32;
  METAL_CONST ushort THREADGROUP_BLOCK_K_FP = gemm_tiling_block_k(GEMM_TILING);
  static_assert(
      THREADGROUP_BLOCK_K_FP % SIMDGROUP_BLOCK_K == 0,
      "FP THREADGROUP_BLOCK_K must be a multiple of SIMDGROUP_BLOCK_K"
  );
  METAL_CONST ushort TILES_M = SIMDGROUP_BLOCK_M / uzu::matmul::MxuFragmentOps::FRAGMENT_ROWS;
  METAL_CONST ushort TILES_N = SIMDGROUP_BLOCK_N / uzu::matmul::MxuFragmentOps::FRAGMENT_COLS;
  METAL_CONST ushort TILES_K = SIMDGROUP_BLOCK_K / uzu::matmul::MxuFragmentOps::FRAGMENT_ROWS;

  METAL_CONST ushort QUANT_BK = (B_PROLOGUE == GemmBPrologueKind::FullPrecision) ? 0 : GROUP_SIZE;
  METAL_CONST ushort PADDING_B = 16 / sizeof(BT);
  METAL_CONST ushort SHARED_STRIDE_B = (QUANT_BK > 0) ? (QUANT_BK + PADDING_B) : 1;
  METAL_CONST ushort THREADGROUP_THREADS = SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * METAL_SIMD_SIZE;
  static_assert(
      B_PROLOGUE == GemmBPrologueKind::FullPrecision || QUANT_BK % SIMDGROUP_BLOCK_K == 0,
      "QUANT_BK must be a multiple of SIMDGROUP_BLOCK_K"
  );
  static_assert(
      B_PROLOGUE == GemmBPrologueKind::FullPrecision || THREADGROUP_BLOCK_K_FP % QUANT_BK == 0,
      "Tile block_k must be a multiple of QUANT_BK"
  );

  using AccumulatorType = float;

  using BLoaderScaleBias = QuantizedBlockLoaderScaleBias<
      BT,
      THREADGROUP_BLOCK_N,
      (QUANT_BK > 0) ? QUANT_BK : 1,
      SHARED_STRIDE_B,
      1,
      THREADGROUP_THREADS,
      (GROUP_SIZE > 0) ? GROUP_SIZE : 1,
      (BITS > 0) ? BITS : 4>;
  using BLoaderScaleZeroPoint = QuantizedBlockLoaderScaleZeroPoint<
      BT,
      THREADGROUP_BLOCK_N,
      (QUANT_BK > 0) ? QUANT_BK : 1,
      SHARED_STRIDE_B,
      1,
      THREADGROUP_THREADS,
      (GROUP_SIZE > 0) ? GROUP_SIZE : 1,
      (BITS > 0) ? BITS : 4>;
  using BLoaderScaleSymmetric = QuantizedBlockLoaderScaleZeroPoint<
      BT,
      THREADGROUP_BLOCK_N,
      (QUANT_BK > 0) ? QUANT_BK : 1,
      SHARED_STRIDE_B,
      1,
      THREADGROUP_THREADS,
      (GROUP_SIZE > 0) ? GROUP_SIZE : 1,
      (BITS > 0) ? BITS : 4,
      true>;

  using AccumFragment = uzu::matmul::Fragment<AccumulatorType, TILES_M, TILES_N, uzu::matmul::MxuFragmentOps>;

  template <bool ALIGNED_M, bool ALIGNED_N, typename Loader>
  static METAL_FUNC AccumFragment quant_k_loop(
      const device AT* a_simdgroup,
      threadgroup BT* b_shared,
      const int leading_dimension_a,
      const int aligned_k_iterations,
      const short simdgroup_limit_m,
      const short simdgroup_limit_n,
      const ushort tile_col_offset,
      const ushort tile_block_cols,
      thread Loader& loader_b,
      const thread ThreadContext& thread_context
  ) {
    AccumFragment accumulator(thread_context);
    accumulator.clear();

    threadgroup BT* b_shared_simdgroup = b_shared + tile_col_offset * SHARED_STRIDE_B;
    const short2 tile_dimensions_b = short2(QUANT_BK, tile_block_cols);

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
      for (int inner_k = 0; inner_k < QUANT_BK; inner_k += SIMDGROUP_BLOCK_K) {
        uzu::matmul::Fragment<AT, TILES_M, TILES_K, uzu::matmul::MxuFragmentOps> left_tile(thread_context);
        uzu::matmul::Fragment<BT, TILES_N, TILES_K, uzu::matmul::MxuFragmentOps> right_tile(thread_context);

        const int left_offset = inner_k;
        if constexpr (ALIGNED_M) {
          left_tile.load(a_simdgroup + left_offset, leading_dimension_a);
        } else {
          left_tile
              .load_safe(a_simdgroup + left_offset, leading_dimension_a, short2(SIMDGROUP_BLOCK_K, simdgroup_limit_m));
        }

        right_tile.load(b_shared_simdgroup + inner_k, int(SHARED_STRIDE_B));

        uzu::matmul::MxuFragmentOps::template tile_matmul<false, true>(accumulator, left_tile, right_tile);
      }

      a_simdgroup += QUANT_BK;
      loader_b.next();
    }

    return accumulator;
  }

  static METAL_FUNC void run(
      const device AT* a,
      const device BT* b,
      device DT* d,
      const constant uzu::matmul::GemmParams* params,
      GemmAlignment alignment,
      GemmDTransform output_transform,
      const device BT* scales,
      const device BT* biases,
      const device uint8_t* zero_points,
      const device BT* output_bias,
      const device int32_t* rht_factors,
      threadgroup BT* b_shared,
      const thread ThreadContext& thread_context
  ) {
    const uint partition = thread_context.threadgroup_position.z;
    const uint tile_y = thread_context.threadgroup_position.y;

    const uint2 tile = tile_id(uint2(thread_context.threadgroup_position.x, tile_y), params);
    const auto geometry = ThreadgroupTileGeometry<THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N>::compute(tile, params);
    if (geometry.out_of_bounds) {
      return;
    }

    const size_t block_row = size_t(geometry.block_row_start);
    const size_t block_col = size_t(geometry.block_col_start);

    const uint k_offset_per_block =
        (B_PROLOGUE == GemmBPrologueKind::FullPrecision) ? uint(THREADGROUP_BLOCK_K_FP) : uint(QUANT_BK);
    const uint k_offset = partition * params->aligned_inner_iterations * k_offset_per_block;

    const device AT* a_block = a + block_row * params->leading_dimension_a + k_offset;
    const device BT* b_block_fp = b + (TRANSPOSE_B ? block_col * params->leading_dimension_b : block_col) +
                                  (TRANSPOSE_B ? k_offset : k_offset * uint(params->leading_dimension_b));

    const ushort tile_row_offset = SIMDGROUP_BLOCK_M * (thread_context.simdgroup_index / SIMDGROUPS_PER_COLUMN);
    const ushort tile_col_offset = SIMDGROUP_BLOCK_N * (thread_context.simdgroup_index % SIMDGROUPS_PER_COLUMN);

    device DT* d_simdgroup = d + size_t(partition) * size_t(params->M) * size_t(params->N) +
                             block_row * params->leading_dimension_d + block_col +
                             tile_row_offset * params->leading_dimension_d + tile_col_offset;

    const short simdgroup_limit_m =
        alignment.contains(GemmAlignment::M)
            ? SIMDGROUP_BLOCK_M
            : short(min(int(SIMDGROUP_BLOCK_M), int(params->M) - int(geometry.block_row_start + tile_row_offset)));
    const short simdgroup_limit_n =
        alignment.contains(GemmAlignment::N)
            ? SIMDGROUP_BLOCK_N
            : short(min(int(SIMDGROUP_BLOCK_N), int(params->N) - int(geometry.block_col_start + tile_col_offset)));

    const device AT* a_simdgroup = a_block + size_t(tile_row_offset) * params->leading_dimension_a;
    const device BT* b_simdgroup_fp =
        b_block_fp +
        (TRANSPOSE_B ? size_t(tile_col_offset) * int(params->leading_dimension_b) : size_t(tile_col_offset));

    const ushort tile_block_cols =
        ushort(min(int(THREADGROUP_BLOCK_N), int(params->N) - int(geometry.block_col_start)));

    const bool apply_scale = output_transform.contains(GemmDTransform::SCALE);
    const bool apply_accumulate = output_transform.contains(GemmDTransform::ACCUMULATE);
    const bool apply_bias = output_transform.contains(GemmDTransform::BIAS);

    const device BT* bias_simdgroup = output_bias + size_t(block_col) + size_t(tile_col_offset);
    using FragType = uzu::matmul::Fragment<DT, TILES_M, TILES_N, uzu::matmul::MxuFragmentOps>;
    const short2 thread_position = FragType::get_position(thread_context);

    auto dispatch_aligned_k = [&](auto body) {
      if constexpr (B_PROLOGUE == GemmBPrologueKind::FullPrecision) {
        dispatch_bool(alignment.contains(GemmAlignment::K), body);
      } else {
        body(true_type{});
      }
    };
    dispatch_aligned_k([&](auto aligned_k) {
      dispatch_bool(
          alignment.contains(GemmAlignment::M) || (simdgroup_limit_m == SIMDGROUP_BLOCK_M),
          [&](auto aligned_m) {
            dispatch_bool(
                alignment.contains(GemmAlignment::N) || (simdgroup_limit_n == SIMDGROUP_BLOCK_N),
                [&](auto aligned_n) {
                  auto accumulator_tile = [&]() {
                    if constexpr (B_PROLOGUE == GemmBPrologueKind::FullPrecision) {
                      const int aligned_k_iterations_fp = int(params->aligned_inner_iterations);
                      return uzu::matmul::gemm_loop<
                          AT,
                          BT,
                          SIMDGROUP_BLOCK_M,
                          SIMDGROUP_BLOCK_N,
                          SIMDGROUP_BLOCK_K,
                          THREADGROUP_BLOCK_K_FP,
                          false,
                          TRANSPOSE_B,
                          aligned_m.value,
                          aligned_n.value,
                          aligned_k.value,
                          AccumulatorType>(
                          a_simdgroup,
                          b_simdgroup_fp,
                          int(params->leading_dimension_a),
                          int(params->leading_dimension_b),
                          int(params->K),
                          aligned_k_iterations_fp,
                          simdgroup_limit_m,
                          simdgroup_limit_n,
                          thread_context
                      );
                    } else {
                      const int aligned_k_iterations_q = int(params->aligned_inner_iterations);
                      constexpr int pack_factor = get_pack_factor<BITS, 8>();
                      constexpr int bytes_per_pack = get_bytes_per_pack<BITS>();
                      const int k_elements = int(params->K);
                      const int weights_row_stride_bytes = k_elements * bytes_per_pack / pack_factor;
                      const int groups_per_row = (k_elements + GROUP_SIZE - 1) / GROUP_SIZE;
                      const int k_offset_groups = int(k_offset) / GROUP_SIZE;
                      const device uint8_t* weights_block = reinterpret_cast<const device uint8_t*>(b) +
                                                            block_col * weights_row_stride_bytes +
                                                            int(k_offset) * bytes_per_pack / pack_factor;
                      const device BT* scales_offset = scales + block_col * groups_per_row + k_offset_groups;

                      auto loader_b = [&]() {
                        if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
                          const device BT* biases_offset = biases + block_col * groups_per_row + k_offset_groups;
                          return BLoaderScaleBias(
                              weights_block,
                              scales_offset,
                              biases_offset,
                              k_elements,
                              b_shared,
                              thread_context.simdgroup_index,
                              thread_context.simd_lane_id
                          );
                        } else if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant) {
                          const int zero_point_stride_per_row =
                              (BITS == 4) ? ((groups_per_row + 1) / 2) : groups_per_row;
                          const device uint8_t* zero_points_row_start =
                              zero_points + block_col * zero_point_stride_per_row +
                              ((BITS == 4) ? (k_offset_groups / 2) : k_offset_groups);
                          return BLoaderScaleZeroPoint(
                              weights_block,
                              scales_offset,
                              zero_points_row_start,
                              k_elements,
                              groups_per_row,
                              b_shared,
                              thread_context.simdgroup_index,
                              thread_context.simd_lane_id
                          );
                        } else {
                          return BLoaderScaleSymmetric(
                              weights_block,
                              scales_offset,
                              nullptr,
                              k_elements,
                              groups_per_row,
                              b_shared,
                              thread_context.simdgroup_index,
                              thread_context.simd_lane_id
                          );
                        }
                      }();

                      return quant_k_loop<aligned_m.value, aligned_n.value>(
                          a_simdgroup,
                          b_shared,
                          int(params->leading_dimension_a),
                          aligned_k_iterations_q,
                          simdgroup_limit_m,
                          simdgroup_limit_n,
                          tile_col_offset,
                          tile_block_cols,
                          loader_b,
                          thread_context
                      );
                    }
                  }();

                  if (apply_scale) {
                    const AccumulatorType scale = AccumulatorType(params->ab_scale);
                    METAL_PRAGMA_UNROLL
                    for (ushort i = 0; i < accumulator_tile.ELEMENTS_PER_TILE; i++) {
                      accumulator_tile.elements()[i] *= scale;
                    }
                  }

                  if (apply_accumulate) {
                    uzu::matmul::Fragment<DT, TILES_M, TILES_N, uzu::matmul::MxuFragmentOps> existing_output(
                        thread_context
                    );
                    if constexpr (aligned_m.value && aligned_n.value) {
                      existing_output.load(d_simdgroup, int(params->leading_dimension_d));
                    } else {
                      existing_output.load_safe(
                          d_simdgroup,
                          int(params->leading_dimension_d),
                          short2(simdgroup_limit_n, simdgroup_limit_m)
                      );
                    }
                    METAL_PRAGMA_UNROLL
                    for (ushort i = 0; i < accumulator_tile.ELEMENTS_PER_TILE; i++) {
                      accumulator_tile.elements()[i] += AccumulatorType(existing_output.elements()[i]);
                    }
                  }

                  if (apply_bias) {
                    METAL_PRAGMA_UNROLL
                    for (ushort tile_row = 0; tile_row < TILES_M; ++tile_row) {
                      METAL_PRAGMA_UNROLL
                      for (ushort tile_col = 0; tile_col < TILES_N; ++tile_col) {
                        thread auto& frag = accumulator_tile.fragment_at(tile_row, tile_col);
                        const short col_base =
                            short(tile_col * uzu::matmul::MxuFragmentOps::FRAGMENT_COLS + thread_position.x);
                        METAL_PRAGMA_UNROLL
                        for (ushort i = 0; i < uzu::matmul::MxuFragmentOps::THREAD_ELEMENT_ROWS; ++i) {
                          METAL_PRAGMA_UNROLL
                          for (ushort j = 0; j < uzu::matmul::MxuFragmentOps::THREAD_ELEMENT_COLS; ++j) {
                            const ushort element_index = i * uzu::matmul::MxuFragmentOps::THREAD_ELEMENT_COLS + j;
                            const short col_local = short(col_base + j);
                            if constexpr (aligned_n.value) {
                              frag[element_index] += AccumulatorType(bias_simdgroup[col_local]);
                            } else {
                              if (col_local < simdgroup_limit_n) {
                                frag[element_index] += AccumulatorType(bias_simdgroup[col_local]);
                              }
                            }
                          }
                        }
                      }
                    }
                  }

                  if constexpr (aligned_m.value && aligned_n.value) {
                    accumulator_tile.store(d_simdgroup, int(params->leading_dimension_d));
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
      device DT* d_block = d + block_row * params->leading_dimension_d + block_col;
      const ushort tile_block_rows = ushort(min(int(THREADGROUP_BLOCK_M), int(params->M) - int(block_row)));
      const ushort tile_block_cols = ushort(min(int(THREADGROUP_BLOCK_N), int(params->N) - int(block_col)));
      apply_output_random_hadamard_transform(
          d_block,
          rht_factors + block_col,
          tile_block_rows,
          tile_block_cols,
          params->leading_dimension_d,
          ushort(SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN),
          thread_context
      );
    }
  }
};

} // namespace gemm
} // namespace uzu
