#pragma once

#include "../../../common/integral_constant.h"
#include "../../../common/thread_context.h"
#include "../../../hadamard_transform/hadamard_transform.h"
#include "../../common/fragment.h"
#include "gemm_rht.h"
#include "../../common/mxu_fragment/ops.h"
#include "../../../generated/matmul.h"
#include "../generated/gemm.h"
#include "block_geometry.h"
#include "gemm_tiling.h"
#include "operands.h"
#include "schedules/dense.h"
#include "schedules/integer.h"
#include "schedules/staged.h"
#include "schedules/tile_context.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <
    typename OutputElementType_,
    GemmTiling GEMM_TILING,
    bool TRANSPOSE_B,
    typename LeftOperand,
    typename RightOperand_>
struct MxuMmaCore {
  using OutputElementType = OutputElementType_;
  using RightOperand = RightOperand_;
  using Left = LeftOperand;
  using Right = RightOperand;
  using LeftElementType = typename Left::ElementType;
  using RightElementType = typename Right::ElementType;
  using LeftStorage = operands::LeftStorage<Left>;
  using RightStorage = operands::RightStorage<Right>;
  using FragmentOps = uzu::matmul::MxuFragmentOps<>;
  using Schedule = metal::conditional_t<
      !Left::quantized,
      metal::conditional_t<!Right::quantized, schedules::DenseSchedule, schedules::StagedSchedule>,
      schedules::IntegerSchedule<Left, Right>>;
  UZU_CONST GemmTiling TILING = GEMM_TILING;
  UZU_CONST ushort THREADGROUP_BLOCK_M = gemm_tiling_block_m(GEMM_TILING);
  UZU_CONST ushort THREADGROUP_BLOCK_N = gemm_tiling_block_n(GEMM_TILING);
  UZU_CONST ushort SIMDGROUPS_PER_ROW = gemm_tiling_simdgroups_per_row(GEMM_TILING);
  UZU_CONST ushort SIMDGROUPS_PER_COLUMN = gemm_tiling_simdgroups_per_column(GEMM_TILING);
  UZU_CONST ushort SIMDGROUP_BLOCK_M = THREADGROUP_BLOCK_M / SIMDGROUPS_PER_ROW;
  UZU_CONST ushort SIMDGROUP_BLOCK_N = THREADGROUP_BLOCK_N / SIMDGROUPS_PER_COLUMN;
  UZU_CONST ushort SIMDGROUP_BLOCK_K = static_cast<ushort>(MXU_SIMDGROUP_BLOCK_K);
  UZU_CONST bool TRANSPOSE_RIGHT = TRANSPOSE_B;
  UZU_CONST ushort THREADGROUP_BLOCK_K_FP = gemm_tiling_block_k(GEMM_TILING);
  static_assert(
      THREADGROUP_BLOCK_K_FP % SIMDGROUP_BLOCK_K == 0,
      "FP THREADGROUP_BLOCK_K must be a multiple of SIMDGROUP_BLOCK_K"
  );
  UZU_CONST ushort TILES_M = SIMDGROUP_BLOCK_M / uzu::matmul::MxuFragmentOps<>::FRAGMENT_ROWS;
  UZU_CONST ushort TILES_N = SIMDGROUP_BLOCK_N / uzu::matmul::MxuFragmentOps<>::FRAGMENT_COLS;
  UZU_CONST ushort TILES_K = SIMDGROUP_BLOCK_K / uzu::matmul::MxuFragmentOps<>::FRAGMENT_ROWS;

  UZU_CONST ushort THREADGROUP_BLOCK_K = Right::template outer_block_k<THREADGROUP_BLOCK_K_FP>();
  UZU_CONST ushort SHARED_STRIDE_B = THREADGROUP_BLOCK_K + 16 / sizeof(RightElementType);
  UZU_CONST ushort THREADGROUP_THREADS = SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * METAL_SIMD_SIZE;

  using AccumulatorType = float;

  using AccumFragment = uzu::matmul::Fragment<AccumulatorType, TILES_M, TILES_N, FragmentOps>;

  static METAL_FUNC void run(
      LeftStorage left,
      RightStorage right,
      device OutputElementType* d,
      const constant uzu::matmul::GemmParams* params,
      GemmAlignment alignment,
      GemmDTransform output_transform,
      const device RightElementType* output_bias,
      const device int32_t* rht_factors,
      threadgroup RightElementType* b_shared,
      const bool stage_scale_lines,
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

    const uint k_offset = partition * params->aligned_inner_iterations * uint(THREADGROUP_BLOCK_K);

    const ushort tile_row_offset = SIMDGROUP_BLOCK_M * (thread_context.simdgroup_index / SIMDGROUPS_PER_COLUMN);
    const ushort tile_col_offset = SIMDGROUP_BLOCK_N * (thread_context.simdgroup_index % SIMDGROUPS_PER_COLUMN);

    device OutputElementType* d_simdgroup = d + size_t(partition) * size_t(params->M) * size_t(params->N) +
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

    schedules::TileContext tile_context;
    tile_context.simdgroup_limit_m = simdgroup_limit_m;
    tile_context.simdgroup_limit_n = simdgroup_limit_n;
    tile_context.block_col = block_col;
    tile_context.tile_col_offset = tile_col_offset;
    tile_context.tile_block_cols =
        ushort(min(int(THREADGROUP_BLOCK_N), int(params->N) - int(geometry.block_col_start)));
    tile_context.k_offset = k_offset;
    tile_context.abs_row_base = uint(geometry.block_row_start) + tile_row_offset;

    const bool apply_scale = output_transform.contains(GemmDTransform::SCALE);
    const bool apply_accumulate = output_transform.contains(GemmDTransform::ACCUMULATE);
    const bool apply_bias = output_transform.contains(GemmDTransform::BIAS);

    const device RightElementType* bias_simdgroup = output_bias + size_t(block_col) + size_t(tile_col_offset);

    dispatch_bool(
        alignment.contains(GemmAlignment::M) || (simdgroup_limit_m == SIMDGROUP_BLOCK_M),
        [&](auto aligned_m) {
          dispatch_bool(
              alignment.contains(GemmAlignment::N) || (simdgroup_limit_n == SIMDGROUP_BLOCK_N),
              [&](auto aligned_n) {
                AccumFragment accumulator_tile;
                dispatch_bool(stage_scale_lines, [&](auto stage) {
                  accumulator_tile =
                      Schedule::template launch<MxuMmaCore, aligned_m.value, aligned_n.value, stage.value>(
                          left,
                          right,
                          b_shared,
                          params,
                          tile_context,
                          alignment,
                          thread_context
                      );
                });

                if (apply_scale) {
                  const AccumulatorType scale = AccumulatorType(params->ab_scale);
                  accumulator_tile.map([&](auto value) { return value * scale; });
                }

                if (apply_accumulate) {
                  uzu::matmul::Fragment<OutputElementType, TILES_M, TILES_N, uzu::matmul::MxuFragmentOps<>>
                      existing_output;
                  auto output_src = uzu::matmul::fragment_source(d_simdgroup, int(params->leading_dimension_d));
                  if constexpr (!(aligned_m.value && aligned_n.value)) {
                    output_src = output_src.bounded(simdgroup_limit_m, simdgroup_limit_n);
                  }
                  existing_output.load_from(thread_context.simd_lane_id, output_src);
                  thread OutputElementType* existing_data = existing_output.elements();
                  accumulator_tile.map([&](auto value) { return value + AccumulatorType(*(existing_data++)); });
                }

                if (apply_bias) {
                  accumulator_tile.map_coords(thread_context.simd_lane_id, [&](short, short col, auto value) {
                    if constexpr (aligned_n.value) {
                      return value + AccumulatorType(bias_simdgroup[col]);
                    } else {
                      if (col < simdgroup_limit_n) {
                        return value + AccumulatorType(bias_simdgroup[col]);
                      }
                      return value;
                    }
                  });
                }

                if constexpr (aligned_m.value && aligned_n.value) {
                  accumulator_tile.store(thread_context.simd_lane_id, d_simdgroup, int(params->leading_dimension_d));
                } else {
                  accumulator_tile.store_safe(
                      thread_context.simd_lane_id,
                      d_simdgroup,
                      int(params->leading_dimension_d),
                      short2(simdgroup_limit_n, simdgroup_limit_m)
                  );
                }
              }
          );
        }
    );

    if (output_transform.contains(GemmDTransform::RHT)) {
      threadgroup_barrier(mem_flags::mem_device);
      device OutputElementType* d_block = d + block_row * params->leading_dimension_d + block_col;
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
