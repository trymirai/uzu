#pragma once

#include "../../../common/integral_constant.h"
#include "../../../common/thread_context.h"
#include "../../../hadamard_transform/hadamard_transform.h"
#include "../../common/fragment.h"
#include "gemm_rht.h"
#include "../../common/mxu_fragment/ops.h"
#include "../../common/mxu_gemm_loop.h"
#include "../../../generated/matmul.h"
#include "../generated/gemm.h"
#include "block_geometry.h"
#include "gemm_tiling.h"
#include "integer_source.h"
#include "../../common/quant_pack.h"
#include "quant_scale_bias.h"
#include "quant_scale_zero_point.h"
#include "../../common/quant_unpack.h"
#include "quantized/metadata.h"
#include "quantized/loader.h"
#include "quantized/slice.h"
#include "quantized/source.h"
#include "schedules/selector.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <
    typename LeftElementType,
    typename RightElementType,
    typename OutputElementType,
    GemmTiling GEMM_TILING,
    bool TRANSPOSE_B,
    GemmBPrologueKind B_PROLOGUE = GemmBPrologueKind::FullPrecision,
    ushort BITS = 0,
    ushort GROUP_SIZE = 0,
    GemmAPrologueKind A_PROLOGUE = GemmAPrologueKind::FullPrecision>
struct MxuMmaCore {
  using LeftType = LeftElementType;
  using RightType = RightElementType;
  using OutputType = OutputElementType;
  using FragmentOps = uzu::matmul::MxuFragmentOps<>;
  using Schedule = typename schedules::ScheduleFor<A_PROLOGUE, B_PROLOGUE, BITS, GROUP_SIZE>::type;
  static_assert(BITS == 0 || BITS == 4 || BITS == 8, "GEMM weight bits must be 0, 4, or 8");
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

  UZU_CONST ushort QUANT_BK = (B_PROLOGUE == GemmBPrologueKind::FullPrecision) ? 0 : GROUP_SIZE;
  UZU_CONST ushort PADDING_B = 16 / sizeof(RightElementType);
  UZU_CONST ushort SHARED_STRIDE_B = (QUANT_BK > 0) ? (QUANT_BK + PADDING_B) : 1;
  UZU_CONST ushort THREADGROUP_THREADS = SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * METAL_SIMD_SIZE;
  static_assert(
      B_PROLOGUE == GemmBPrologueKind::FullPrecision || QUANT_BK % SIMDGROUP_BLOCK_K == 0,
      "QUANT_BK must be a multiple of SIMDGROUP_BLOCK_K"
  );
  static_assert(
      B_PROLOGUE == GemmBPrologueKind::FullPrecision || THREADGROUP_BLOCK_K_FP % QUANT_BK == 0,
      "Tile block_k must be a multiple of QUANT_BK"
  );

  using AccumulatorType = float;

  using AccumFragment = uzu::matmul::Fragment<AccumulatorType, TILES_M, TILES_N, FragmentOps>;

  static METAL_FUNC void run(
      const device LeftElementType* a,
      const device RightElementType* b,
      device OutputElementType* d,
      const constant uzu::matmul::GemmParams* params,
      GemmAlignment alignment,
      GemmDTransform output_transform,
      const bool signed_codes,
      const device RightElementType* scales,
      const device RightElementType* biases,
      const device uint8_t* zero_points,
      const device RightElementType* output_bias,
      const device int32_t* rht_factors,
      const device int8_t* a_int8,
      const device float* a_scales,
      const device int32_t* a_group_sums,
      threadgroup RightElementType* b_shared,
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

    const device RightElementType* b_block_fp = b +
                                                (TRANSPOSE_B ? block_col * params->leading_dimension_b : block_col) +
                                                (TRANSPOSE_B ? k_offset : k_offset * uint(params->leading_dimension_b));

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

    const device LeftElementType* a_simdgroup = a;
    if constexpr (A_PROLOGUE == GemmAPrologueKind::FullPrecision) {
      a_simdgroup +=
          block_row * params->leading_dimension_a + k_offset + size_t(tile_row_offset) * params->leading_dimension_a;
    }
    const device RightElementType* b_simdgroup_fp =
        b_block_fp +
        (TRANSPOSE_B ? size_t(tile_col_offset) * int(params->leading_dimension_b) : size_t(tile_col_offset));

    const ushort tile_block_cols =
        ushort(min(int(THREADGROUP_BLOCK_N), int(params->N) - int(geometry.block_col_start)));

    const bool apply_scale = output_transform.contains(GemmDTransform::SCALE);
    const bool apply_accumulate = output_transform.contains(GemmDTransform::ACCUMULATE);
    const bool apply_bias = output_transform.contains(GemmDTransform::BIAS);

    const device RightElementType* bias_simdgroup = output_bias + size_t(block_col) + size_t(tile_col_offset);

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
                    if constexpr (A_PROLOGUE == GemmAPrologueKind::Int8Symmetric) {
                      using RightFormat = typename Schedule::RightFormat;
                      const auto quantized_right = make_quantized_slice<RightFormat, Schedule::RIGHT_GROUP_SIZE>(
                          b,
                          block_col,
                          k_offset,
                          int(params->K)
                      );
                      const device int8_t* a_int8_simdgroup = a_int8 + block_row * params->leading_dimension_a +
                                                              k_offset +
                                                              size_t(tile_row_offset) * params->leading_dimension_a;
                      const device uint8_t* b_packed_simdgroup =
                          quantized_right.storage + size_t(tile_col_offset) * quantized_right.row_stride_bytes;
                      return Schedule::template run<MxuMmaCore, aligned_m.value, aligned_n.value>(
                          a_int8_simdgroup,
                          b_packed_simdgroup,
                          a_scales,
                          a_group_sums,
                          scales,
                          biases,
                          zero_points,
                          int(params->leading_dimension_a),
                          quantized_right.row_stride_bytes,
                          int(params->aligned_inner_iterations),
                          simdgroup_limit_m,
                          simdgroup_limit_n,
                          uint(geometry.block_row_start) + tile_row_offset,
                          uint(geometry.block_col_start) + tile_col_offset,
                          uint(quantized_right.first_group),
                          k_offset / uint(Schedule::LEFT_GROUP_SIZE),
                          uint(quantized_right.groups_per_row),
                          uint(params->K) / uint(Schedule::LEFT_GROUP_SIZE),
                          thread_context
                      );
                    } else if constexpr (B_PROLOGUE == GemmBPrologueKind::FullPrecision) {
                      const int aligned_k_iterations_fp = int(params->aligned_inner_iterations);
                      return Schedule::template run<MxuMmaCore, aligned_m.value, aligned_n.value, aligned_k.value>(
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
                      using RightFormat = typename Schedule::RightFormat;
                      const int aligned_k_iterations_q = int(params->aligned_inner_iterations);
                      const int k_elements = int(params->K);
                      const auto quantized_right = make_quantized_slice<RightFormat, Schedule::RIGHT_GROUP_SIZE>(
                          b,
                          block_col,
                          k_offset,
                          k_elements
                      );
                      const int groups_per_row = quantized_right.groups_per_row;
                      const int k_offset_groups = quantized_right.first_group;
                      const device uint8_t* right_storage = quantized_right.storage;
                      const device RightElementType* scales_offset =
                          scales + block_col * groups_per_row + k_offset_groups;

                      using Loaders = QuantizedLoaders<
                          RightElementType,
                          THREADGROUP_BLOCK_N,
                          (QUANT_BK > 0) ? QUANT_BK : 1,
                          SHARED_STRIDE_B,
                          1,
                          THREADGROUP_THREADS,
                          Schedule::RIGHT_GROUP_SIZE,
                          RightFormat::BITS>;

                      auto loader_b = [&]() {
                        if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
                          const device RightElementType* biases_offset =
                              biases + block_col * groups_per_row + k_offset_groups;
                          return make_loader<B_PROLOGUE, typename Loaders::ScaleBias>(
                              right_storage,
                              scales_offset,
                              biases_offset,
                              nullptr,
                              signed_codes,
                              k_elements,
                              groups_per_row,
                              b_shared,
                              thread_context.simdgroup_index,
                              thread_context.simd_lane_id
                          );
                        } else if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant) {
                          const int zero_point_stride_per_row =
                              zero_point_row_stride<RightFormat::BITS>(groups_per_row);
                          const device uint8_t* zero_points_row_start =
                              zero_points + block_col * zero_point_stride_per_row +
                              ((RightFormat::BITS == 4) ? (k_offset_groups / 2) : k_offset_groups);
                          return make_loader<B_PROLOGUE, typename Loaders::ScaleZeroPoint>(
                              right_storage,
                              scales_offset,
                              static_cast<const device RightElementType*>(nullptr),
                              zero_points_row_start,
                              signed_codes,
                              k_elements,
                              groups_per_row,
                              b_shared,
                              thread_context.simdgroup_index,
                              thread_context.simd_lane_id
                          );
                        } else if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleSymmetricDequant) {
                          return make_loader<
                              GemmBPrologueKind::ScaleSymmetricDequant,
                              typename Loaders::ScaleSymmetric>(
                              right_storage,
                              scales_offset,
                              static_cast<const device RightElementType*>(nullptr),
                              nullptr,
                              signed_codes,
                              k_elements,
                              groups_per_row,
                              b_shared,
                              thread_context.simdgroup_index,
                              thread_context.simd_lane_id
                          );
                        }
                      }();

                      return Schedule::template run<MxuMmaCore, aligned_m.value, aligned_n.value>(
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
    });

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
