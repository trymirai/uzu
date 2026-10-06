#include "../../common/dsl.h"
#include "../../common/defines.h"
#include "../../common/thread_context.h"
#include "../generated/gemm.h"

#include "common/gemm_tiling.h"
#include "common/mxu_mma_core.h"
#include "common/quantized/trellis.h"

using namespace metal;
using namespace uzu::gemm;

#define REDUCE_THREADS 256
#define REDUCE_VECTOR_WIDTH 4

namespace {

struct TrellisEpilogue {
  const device float* scale_and_offsets;
  const device float4* group_sums;
  const device float* activation_scales;
  const device float* row_scales;
  device bfloat* output;
  uint output_row_stride;

  METAL_FUNC void store(const int biased_dot, const uint token, const uint row) const {
    const float4 group_sum = group_sums[token];
    const int level_dot =
        biased_dot - int(trellis::WEIGHT_BIAS) * int(group_sum.x + group_sum.y + group_sum.z + group_sum.w);
    const float activation_scale = activation_scales[token];
    const float4 offsets =
        float4(scale_and_offsets[1], scale_and_offsets[2], scale_and_offsets[3], scale_and_offsets[4]);
    const float dot = float(level_dot) * scale_and_offsets[0] + metal::dot(group_sum, offsets);
    output[token * output_row_stride + row] = bfloat(dot * row_scales[row] * activation_scale);
  }
};

} // namespace

template <GemmTiling GEMM_TILING>
VARIANTS(
    GEMM_TILING,
    GemmTiling::Tile16x32x256_Simdgroups1x1,
    GemmTiling::Tile64x64x256_Simdgroups2x2,
    GemmTiling::Tile128x128x256_Simdgroups4x4)
KERNEL(GemmTrellis)(
    const device int8_t* activations,
    const device float4* column_group_sums,
    const device float* activation_scales,
    const device uint8_t* codes,
    const device float* row_scales,
    const device float* scale_and_offsets,
    device bfloat* output,
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    const GemmAlignment alignment SPECIALIZE,
    const uint trellis_vector_width SPECIALIZE,
    const uint trellis_transition_bits SPECIALIZE,
    const uint trellis_restart_columns SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(METAL_SIMD_SIZE),
    const uint thread_y THREADS(gemm_tiling_simdgroups_per_column(GEMM_TILING)),
    const uint thread_z THREADS(gemm_tiling_simdgroups_per_row(GEMM_TILING)),
    const ThreadContext thread_context
) {
  (void)thread_x;
  (void)thread_y;
  (void)thread_z;

  using LeftOperand =
      operands::LeftOperandFor<GemmAPrologueKind::Int8Symmetric, bfloat, ushort(trellis::K_STEP), false>;
  // A stock int8 right operand, only for the tile constants of `Core`.
  using RightOperand =
      operands::RightOperandFor<GemmBPrologueKind::ScaleSymmetricDequant, ushort(8), ushort(trellis::K_STEP), bfloat>;
  using Core = MxuMmaCore<bfloat, GEMM_TILING, true, LeftOperand, RightOperand>;
  using Ops = typename Core::FragmentOps;
  using Products = uzu::matmul::Fragment<int, Core::TILES_M, Core::TILES_N, Ops>;
  using RightFragment = uzu::matmul::Fragment<int8_t, Core::TILES_N, Core::TILES_K, Ops, uzu::matmul::ReadDirect, true>;
  const trellis::GemmTrellisFormat trellis_format{
      trellis_vector_width,
      trellis_transition_bits,
      trellis_restart_columns
  };

  const uint simdgroup = thread_context.simdgroup_index;
  const uint token_base =
      group_y * Core::THREADGROUP_BLOCK_M + Core::SIMDGROUP_BLOCK_M * (simdgroup / Core::SIMDGROUPS_PER_COLUMN);
  const uint row_base =
      group_x * Core::THREADGROUP_BLOCK_N + Core::SIMDGROUP_BLOCK_N * (simdgroup % Core::SIMDGROUPS_PER_COLUMN);
  const short token_end = short(min(int(Core::SIMDGROUP_BLOCK_M), int(params->M) - int(token_base)));
  const short row_end = short(min(int(Core::SIMDGROUP_BLOCK_N), int(params->N) - int(row_base)));
  const schedules::TileContext tile_context{
      .simdgroup_limit_m = token_end,
      .simdgroup_limit_n = row_end,
      .k_offset = group_z * params->aligned_inner_iterations * trellis::K_STEP,
      .abs_row_base = token_base,
  };

  const uint code_row_bytes = trellis::row_bytes(trellis_format, params->K);
  const short2 position = Ops::get_position(thread_context.simd_lane_id);
  const auto left_storage = operands::pack_left<LeftOperand, bfloat>(nullptr, activations, nullptr, nullptr);

  Products products;
  products.clear();
  uzu::dispatch_bool(alignment.contains(GemmAlignment::M) || token_end == Core::SIMDGROUP_BLOCK_M, [&](auto aligned_m) {
    uzu::dispatch_bool(alignment.contains(GemmAlignment::N) || row_end == Core::SIMDGROUP_BLOCK_N, [&](auto aligned_n) {
      auto left = quantized::make_left_cursor<true, Core, LeftOperand, aligned_m.value>(
          left_storage,
          params,
          tile_context,
          thread_context
      );
      quantized::TrellisCursor<RightFragment, aligned_n.value> trellis_cursor{
          codes + size_t(row_base + position.y) * code_row_bytes,
          code_row_bytes,
          row_end,
          position,
          tile_context.k_offset,
          trellis_format
      };
      const int k_chunks = token_end > 0 && row_end > 0
                               ? int(params->aligned_inner_iterations) * int(trellis::K_STEP / Core::SIMDGROUP_BLOCK_K)
                               : 0;
      METAL_PRAGMA_NO_UNROLL
      for (int chunk = 0; chunk < k_chunks; ++chunk) {
        auto left_tile = left.load(0u);
        auto right_tile = trellis_cursor.load(0u);
        uzu::matmul::fragment_mma(products, left_tile, right_tile);
        left.advance();
        trellis_cursor.advance();
      }
    });
  });

  const bool write_split_k_partials = params->aligned_inner_iterations * trellis::K_STEP < params->K;
  const size_t partial_sum_offset = size_t(group_z) * params->M * params->N;
  const TrellisEpilogue epilogue{
      scale_and_offsets,
      column_group_sums,
      activation_scales,
      row_scales,
      output,
      params->leading_dimension_d
  };
  products.map_coords(thread_context.simd_lane_id, [&](short token_offset, short row_offset, int value) {
    if (token_offset < token_end && row_offset < row_end) {
      const uint token = token_base + uint(token_offset);
      const uint row = row_base + uint(row_offset);
      if (write_split_k_partials) {
        device int* partial_sums = reinterpret_cast<device int*>(output) + partial_sum_offset;
        partial_sums[size_t(token) * params->N + row] = value;
      } else {
        epilogue.store(value, token, row);
      }
    }
    return value;
  });
}

KERNEL(GemmTrellisReduce)(
    const device int4* partial_sums,
    const device float4* column_group_sums,
    const device float* activation_scales,
    const device float* row_scales,
    const device float* scale_and_offsets,
    device bfloat* output,
    const constant uint& token_count,
    const constant uint& row_count,
    const constant uint& partition_count,
    const constant uint& output_stride,
    const uint threadgroup_index GROUPS((token_count * (row_count / REDUCE_VECTOR_WIDTH)).div_ceil(REDUCE_THREADS)),
    const uint thread_index_in_threadgroup THREADS(REDUCE_THREADS)
) {
  const uint row_groups = row_count / REDUCE_VECTOR_WIDTH;
  const uint token_row_group = threadgroup_index * REDUCE_THREADS + thread_index_in_threadgroup;
  if (token_row_group >= token_count * row_groups) {
    return;
  }
  int4 partial_sum = int4(0);
  for (uint partition = 0u; partition < partition_count; ++partition) {
    partial_sum += partial_sums[partition * token_count * row_groups + token_row_group];
  }
  const uint token = token_row_group / row_groups;
  const uint first_row = token_row_group % row_groups * REDUCE_VECTOR_WIDTH;
  const TrellisEpilogue
      epilogue{scale_and_offsets, column_group_sums, activation_scales, row_scales, output, output_stride};
  for (uint index = 0u; index < REDUCE_VECTOR_WIDTH; ++index) {
    epilogue.store(partial_sum[index], token, first_row + index);
  }
}
