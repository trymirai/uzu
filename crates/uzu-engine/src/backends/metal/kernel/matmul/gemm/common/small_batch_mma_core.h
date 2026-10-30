#pragma once

#include <metal_stdlib>

#include "../../../common/thread_context.h"
#include "../../../generated/matmul.h"
#include "../generated/gemm.h"
#include "../../../common/soft_cap.h"
#include "../../common/fragment.h"
#include "../../common/simdgroup_fragment_ops.h"
#include "block_geometry.h"
#include "gemm_rht.h"
#include "gemm_tiling.h"
#include "operands.h"
#include "schedules/small_batch/packed.h"
#include "schedules/small_batch/direct.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <typename OutputElementType_, GemmTiling GEMM_TILING, typename LeftOperand, typename RightOperand>
struct SmallBatchMmaCore {
  using OutputElementType = OutputElementType_;
  using Left = LeftOperand;
  using Right = RightOperand;
  using RightElementType = typename Right::ElementType;
  using LeftStorage = operands::LeftStorage<Left>;
  using RightStorage = operands::RightStorage<Right>;
  using Schedule = metal::conditional_t<Left::PACKED_HALF, schedules::PackedSchedule, schedules::DirectSchedule>;
  using FragmentOps = uzu::matmul::SimdgroupFragmentOps;
  UZU_CONST int THREADGROUP_BLOCK_M = gemm_tiling_block_m(GEMM_TILING);
  UZU_CONST int THREADGROUP_BLOCK_N = gemm_tiling_block_n(GEMM_TILING);
  UZU_CONST int THREADGROUP_BLOCK_K = gemm_tiling_block_k(GEMM_TILING);
  UZU_CONST int SIMDGROUPS_PER_ROW = gemm_tiling_simdgroups_per_row(GEMM_TILING);
  UZU_CONST int SIMDGROUPS_PER_COLUMN = gemm_tiling_simdgroups_per_column(GEMM_TILING);
  UZU_CONST int SIMDGROUPS_PER_K = gemm_tiling_simdgroups_per_k(GEMM_TILING);
  UZU_CONST int SIMDGROUP_BLOCK_M = THREADGROUP_BLOCK_M / SIMDGROUPS_PER_ROW;
  UZU_CONST int SIMDGROUP_BLOCK_N = THREADGROUP_BLOCK_N / SIMDGROUPS_PER_COLUMN;
  UZU_CONST ushort TILES_M = Schedule::TRANSPOSED_OUTPUT ? SIMDGROUP_BLOCK_N / FragmentOps::FRAGMENT_ROWS
                                                         : SIMDGROUP_BLOCK_M / FragmentOps::FRAGMENT_ROWS;
  UZU_CONST ushort TILES_N = Schedule::TRANSPOSED_OUTPUT ? SIMDGROUP_BLOCK_M / FragmentOps::FRAGMENT_COLS
                                                         : SIMDGROUP_BLOCK_N / FragmentOps::FRAGMENT_COLS;
  using AccumFragment = uzu::matmul::Fragment<float, TILES_M, TILES_N, FragmentOps>;

  using BlockGeometry = ThreadgroupTileGeometry<THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N>;
  struct TileContext {
    BlockGeometry block_geometry;
    ushort simd_lane_id;
    short2 lane_position;
    uint k_partition;
    ushort simdgroup_k_slice;
    uint simdgroup_column_base;
    uint k_block_begin;
    uint k_block_end;
  };

  static METAL_FUNC TileContext
  make_tile_context(const constant uzu::matmul::GemmParams* params, const thread ThreadContext& thread_context) {
    const ushort simd_lane_id = ushort(thread_context.simd_lane_id);
    const uint2 id =
        tile_id(uint2(thread_context.threadgroup_position.x, thread_context.threadgroup_position.y), params);
    const BlockGeometry block_geometry = BlockGeometry::compute(id, params);
    const ushort simdgroup_index = ushort(thread_context.simdgroup_index);
    const uint total_blocks = params->K / THREADGROUP_BLOCK_K;
    const uint blocks_per_partition =
        SIMDGROUPS_PER_K * params->aligned_inner_iterations + total_blocks % SIMDGROUPS_PER_K;
    const uint k_block_begin = thread_context.threadgroup_position.z * blocks_per_partition;
    return {
        block_geometry,
        simd_lane_id,
        FragmentOps::get_position(simd_lane_id),
        thread_context.threadgroup_position.z,
        ushort(simdgroup_index / SIMDGROUPS_PER_COLUMN),
        block_geometry.block_col_start + uint(simdgroup_index % SIMDGROUPS_PER_COLUMN) * SIMDGROUP_BLOCK_N,
        k_block_begin,
        k_block_begin + blocks_per_partition
    };
  }

  static METAL_FUNC void run(
      LeftStorage left,
      RightStorage right,
      device OutputElementType* d,
      const constant uzu::matmul::GemmParams* params,
      GemmDTransform output_transform,
      const constant float& soft_cap,
      const device RightElementType* output_bias,
      const device int32_t* rht_factors,
      threadgroup RightElementType* scratch,
      const thread ThreadContext& thread_context
  ) {
    TileContext tile_context = make_tile_context(params, thread_context);
    if (tile_context.block_geometry.out_of_bounds) {
      return;
    }
    AccumFragment accumulator = Schedule::template launch<SmallBatchMmaCore>(left, right, params, tile_context);
    const bool output_writer = accumulator.template reduce_k_slices<SIMDGROUPS_PER_K, SIMDGROUPS_PER_COLUMN>(
        reinterpret_cast<threadgroup float*>(scratch),
        thread_context
    );
    const bool needs_rht = output_transform.contains(GemmDTransform::RHT);
    if (!output_writer && !needs_rht) {
      return;
    }

    const uint row_start = tile_context.block_geometry.block_row_start;
    const uint m_remaining = min(uint(THREADGROUP_BLOCK_M), params->M - row_start);
    const uint n_remaining = tile_context.simdgroup_column_base < params->N
                                 ? min(uint(SIMDGROUP_BLOCK_N), params->N - tile_context.simdgroup_column_base)
                                 : 0;
    const uint safe_output_col = min(tile_context.simdgroup_column_base, params->N - 1);
    const short2 tile_dimensions = Schedule::TRANSPOSED_OUTPUT ? short2(short(m_remaining), short(n_remaining))
                                                               : short2(short(n_remaining), short(m_remaining));
    device OutputElementType* row_destination =
        d + size_t(tile_context.k_partition) * size_t(params->M) * size_t(params->N) +
        size_t(row_start) * params->leading_dimension_d;
    if (output_writer) {
      if (output_transform.contains(GemmDTransform::SCALE)) {
        const uzu::matmul::TransformScale<float, float> scale(params->ab_scale);
        accumulator.map([&](float value) { return scale.apply(value); });
      }
      const bool needs_bias = output_transform.contains(GemmDTransform::BIAS);
      if (needs_bias && !needs_rht) {
        accumulator.map_coords(tile_context.simd_lane_id, [&](short row, short col, float value) {
          const uint bias_col = Schedule::TRANSPOSED_OUTPUT ? uint(row) : uint(col);
          return bias_col < n_remaining ? value + float(output_bias[tile_context.simdgroup_column_base + bias_col])
                                        : value;
        });
      }
      if (output_transform.contains(GemmDTransform::SOFT_CAP)) {
        accumulator.map([&](float value) { return uzu::apply_soft_cap(value, soft_cap); });
      }

      device OutputElementType* destination = row_destination + safe_output_col;
      const int row_stride = Schedule::TRANSPOSED_OUTPUT ? 1 : int(params->leading_dimension_d);
      const auto col_stride = [&]() {
        if constexpr (Schedule::TRANSPOSED_OUTPUT) {
          return int(params->leading_dimension_d);
        } else {
          return uzu::Int<1>{};
        }
      }();
      accumulator.store_safe(
          tile_context.simd_lane_id,
          destination,
          row_stride,
          tile_dimensions,
          uzu::Int<1>{},
          uzu::Int<1>{},
          col_stride
      );
    }

    if (needs_rht) {
      const uint tile_col_start = tile_context.block_geometry.block_col_start;
      const ushort tile_block_cols = ushort(min(int(THREADGROUP_BLOCK_N), int(params->N) - int(tile_col_start)));
      device OutputElementType* tile_output = row_destination + tile_col_start;
      const device RightElementType* tile_bias = output_bias;
      if (output_transform.contains(GemmDTransform::BIAS)) {
        tile_bias += tile_col_start;
      }

      threadgroup_barrier(mem_flags::mem_device);
      apply_output_random_hadamard_transform(
          tile_output,
          rht_factors + tile_col_start,
          tile_bias,
          output_transform.contains(GemmDTransform::BIAS),
          ushort(m_remaining),
          tile_block_cols,
          params->leading_dimension_d,
          SIMDGROUPS_PER_K * SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN,
          thread_context
      );
    }
  }
};

} // namespace gemm
} // namespace uzu
