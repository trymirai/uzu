#include "../../common/dsl.h"
#include "../../generated/trellis.h"
#include "../../common/defines.h"
#include "../../common/thread_context.h"
#include "../generated/gemm.h"

#include "common/gemm_tiling.h"
#include "common/mxu_mma_core.h"
#include "common/quantized/cursor.h"
#include "common/quantized/trellis.h"
#include "common/simdgroup_mma_core.h"
#include "../common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::gemm;
using uzu::trellis::TrellisCodebook;

#define REDUCE_THREADS 256
#define REDUCE_VECTOR_WIDTH 4
// Pre-M5 stages both operands as half in threadgroup memory, each row padded by 16 bytes.
#define TRELLIS_STAGE_ELEMENTS                                                                                         \
  (USE_MXU ? 1                                                                                                         \
           : (gemm_tiling_block_m(GEMM_TILING) + gemm_tiling_block_n(GEMM_TILING)) *                                   \
                 (gemm_tiling_block_k(GEMM_TILING) + 8))

namespace {

struct TrellisEpilogue {
  TrellisCodebook codebook;
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
    const float4 offsets = float4(codebook.offsets[0], codebook.offsets[1], codebook.offsets[2], codebook.offsets[3]);
    const float dot = float(level_dot) * codebook.scale + metal::dot(group_sum, offsets);
    output[token * output_row_stride + row] = bfloat(dot * row_scales[row] * activation_scale);
  }
};

struct TrellisWeightLoader {
  trellis::GemmTrellisFormat format;
  const device uint8_t* codes;
  uint row_byte;
  uint row_columns;
  uint column;
  uint row;
  uint block_n;
  threadgroup half* destination;
  uint block_k;

  METAL_FUNC void load_unsafe() const { load(true); }

  METAL_FUNC void load_safe(const short2 valid) const { load(row < uint(valid.y)); }

  METAL_FUNC void next() { column += block_k; }

  METAL_FUNC void load(const bool live) const {
    if (row >= block_n) {
      return;
    }
    const uint4 weights =
        live ? trellis::decode_biased_weight_quad(format, codes, row_byte, column, row_columns) : uint4(0u);
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < 4; ++index) {
      *reinterpret_cast<threadgroup half4*>(destination + 4 * index) = half4(as_type<uchar4>(weights[index]));
    }
  }
};

} // namespace

template <GemmTiling GEMM_TILING, bool USE_MXU>
VARIANTS(
    GEMM_TILING,
    GemmTiling::Tile16x32x32_Simdgroups2x1,
    GemmTiling::Tile32x32x32_Simdgroups2x2,
    GemmTiling::Tile64x64x32_Simdgroups2x2,
    GemmTiling::Tile16x32x256_Simdgroups1x1,
    GemmTiling::Tile32x64x256_Simdgroups2x2,
    GemmTiling::Tile64x64x256_Simdgroups2x2,
    GemmTiling::Tile128x128x256_Simdgroups4x4)
VARIANTS(USE_MXU, false, true)
CONSTRAINT(
    USE_MXU ==
    (GEMM_TILING == GemmTiling::Tile16x32x256_Simdgroups1x1 || GEMM_TILING == GemmTiling::Tile32x64x256_Simdgroups2x2 ||
     GEMM_TILING == GemmTiling::Tile64x64x256_Simdgroups2x2 || GEMM_TILING == GemmTiling::Tile128x128x256_Simdgroups4x4))
KERNEL(GemmTrellis)(
    const device int8_t* activations,
    const device float4* column_group_sums,
    const device float* activation_scales,
    const device uint8_t* codes,
    const device float* row_scales,
    const constant TrellisCodebook& codebook,
    device bfloat* output,
    const constant uzu::matmul::GemmParams* params,
    threadgroup half stage[TRELLIS_STAGE_ELEMENTS],
    const constant uint& partition_count,
    const GemmAlignment alignment SPECIALIZE,
    const uint vector_width SPECIALIZE,
    const uint transition_bits SPECIALIZE,
    const uint restart_columns SPECIALIZE,
    const uint group_x GROUPS(params[0].threadgroups_per_row),
    const uint group_y GROUPS(params[0].threadgroups_per_column),
    const uint group_z GROUPS(partition_count),
    const uint thread_x THREADS(METAL_SIMD_SIZE),
    const uint thread_y THREADS(gemm_tiling_simdgroups_per_column(GEMM_TILING)),
    const uint thread_z THREADS(gemm_tiling_simdgroups_per_row(GEMM_TILING)),
    const ThreadContext thread_context
) {
  (void)thread_x;
  (void)thread_y;
  (void)thread_z;

  using Ops = metal::conditional_t<USE_MXU, uzu::matmul::MxuFragmentOps<>, uzu::matmul::SimdgroupFragmentOps>;
  constexpr ushort BLOCK_M = gemm_tiling_block_m(GEMM_TILING);
  constexpr ushort BLOCK_N = gemm_tiling_block_n(GEMM_TILING);
  constexpr ushort SIMDGROUPS_PER_COLUMN = gemm_tiling_simdgroups_per_column(GEMM_TILING);
  constexpr ushort SIMDGROUP_BLOCK_M = BLOCK_M / gemm_tiling_simdgroups_per_row(GEMM_TILING);
  constexpr ushort SIMDGROUP_BLOCK_N = BLOCK_N / SIMDGROUPS_PER_COLUMN;
  constexpr ushort TILES_M = SIMDGROUP_BLOCK_M / Ops::FRAGMENT_ROWS;
  constexpr ushort TILES_N = SIMDGROUP_BLOCK_N / Ops::FRAGMENT_COLS;
  using Products = uzu::matmul::Fragment<int, TILES_M, TILES_N, Ops>;
  const trellis::GemmTrellisFormat trellis_format{vector_width, transition_bits, restart_columns};

  const uint simdgroup = thread_context.simdgroup_index;
  const uint token_base = group_y * BLOCK_M + SIMDGROUP_BLOCK_M * (simdgroup / SIMDGROUPS_PER_COLUMN);
  const uint row_base = group_x * BLOCK_N + SIMDGROUP_BLOCK_N * (simdgroup % SIMDGROUPS_PER_COLUMN);
  const short token_end = short(min(int(SIMDGROUP_BLOCK_M), int(params->M) - int(token_base)));
  const short row_end = short(min(int(SIMDGROUP_BLOCK_N), int(params->N) - int(row_base)));
  const uint k_offset = group_z * params->aligned_inner_iterations * trellis::K_STEP;

  const uint code_row_bytes = trellis::row_bytes(trellis_format, params->K);
  const short2 position = Ops::get_position(thread_context.simd_lane_id);

  const bool write_split_k_partials = params->aligned_inner_iterations * trellis::K_STEP < params->K;
  const size_t partial_sum_offset = size_t(group_z) * params->M * params->N;
  const TrellisEpilogue
      epilogue{codebook, column_group_sums, activation_scales, row_scales, output, params->leading_dimension_d};
  const auto store = [&](const uint token, const uint row, const int value) {
    if (token < params->M && row < params->N) {
      if (write_split_k_partials) {
        device int* partial_sums = reinterpret_cast<device int*>(output) + partial_sum_offset;
        partial_sums[size_t(token) * params->N + row] = value;
      } else {
        epilogue.store(value, token, row);
      }
    }
  };

  Products products;
  products.clear();
  if constexpr (USE_MXU) {
    constexpr ushort TILES_K = MXU_SIMDGROUP_BLOCK_K / Ops::FRAGMENT_ROWS;
    using RightFragment = uzu::matmul::Fragment<int8_t, TILES_N, TILES_K, Ops, uzu::matmul::ReadDirect, true>;
    // Each lane decodes 16 consecutive K columns: chunk 0 takes columns 0..7, chunk 1 takes 8..15
    using LeftFragment = uzu::matmul::Fragment<int8_t, TILES_M, TILES_K, Ops>;
    const uint k_steps = token_end > 0 && row_end > 0 ? params->aligned_inner_iterations : 0;
    const device int8_t* lane_activations = activations + size_t(token_base + position.y) * params->leading_dimension_a;
    // Rows past M re-read the last row; the store drops them
    const bool aligned_m = alignment.contains(GemmAlignment::M);
    const device int8_t* tile_activations = activations + size_t(token_base) * params->leading_dimension_a;
    const short last_tile_row = token_end - 1;
    const uint lane_row_byte = (row_base + position.y) * code_row_bytes;
    const auto fill = [](thread auto& fragments, ushort fragment_row, ushort row_slot, const uint4 values) {
      METAL_PRAGMA_UNROLL
      for (uint index = 0; index < 4; ++index) {
        reinterpret_cast<thread uint*>(&fragments[index / 2].fragment_at(fragment_row, index % 2))[row_slot] =
            values[index];
      }
    };
    uzu::dispatch_bool(alignment.contains(GemmAlignment::N) || row_end == SIMDGROUP_BLOCK_N, [&](auto aligned_n) {
      METAL_PRAGMA_NO_UNROLL
      for (uint step = 0; step < k_steps; ++step) {
        const uint k = k_offset + step * trellis::K_STEP + 4 * uint(position.x);
        LeftFragment left[2];
        RightFragment right[2];
        quantized::for_each_fragment_row<LeftFragment>([&](ushort fragment_row, ushort row_slot, short row_offset) {
          const device int8_t* row = aligned_m ? lane_activations + row_offset * params->leading_dimension_a
                                               : tile_activations + min(short(position.y + row_offset), last_tile_row) *
                                                                        params->leading_dimension_a;
          fill(left, fragment_row, row_slot, *reinterpret_cast<const device uint4*>(row + k));
        });
        quantized::for_each_fragment_row<RightFragment>([&](ushort fragment_row, ushort row_slot, short row_offset) {
          const bool live = aligned_n.value || position.y + row_offset < row_end;
          const uint4 weights = live ? trellis::decode_biased_weight_quad(
                                           trellis_format,
                                           codes,
                                           lane_row_byte + row_offset * code_row_bytes,
                                           k,
                                           params->K
                                       )
                                     : uint4(0u);
          fill(right, fragment_row, row_slot, weights);
        });
        METAL_PRAGMA_UNROLL
        for (uint chunk = 0; chunk < 2; ++chunk) {
          uzu::matmul::fragment_mma(products, left[chunk], right[chunk]);
        }
      }
    });
    products.map_coords(thread_context.simd_lane_id, [&](short token_offset, short row_offset, int value) {
      store(token_base + token_offset, row_base + row_offset, value);
      return value;
    });
  } else {
    using LeftOperand = operands::LeftOperandFor<GemmAPrologueKind::FullPrecision, half, ushort(0), false>;
    using RightOperand = operands::RightOperandFor<GemmBPrologueKind::FullPrecision, ushort(0), ushort(0), half>;
    using Core = SimdgroupMmaCore<float, GEMM_TILING, true, LeftOperand, RightOperand>;
    constexpr ushort BLOCK_K = Core::THREADGROUP_BLOCK_K;
    constexpr ushort A_READS = BLOCK_M * BLOCK_K / Core::THREADGROUP_THREADS;
    constexpr ushort A_COLS = BLOCK_K / A_READS;
    constexpr uint SLAB_ITERATIONS = 1024 / BLOCK_K;
    static_assert(SLAB_ITERATIONS * BLOCK_K * 128 * 111 < (1 << 24), "a slab must sum exactly in float");
    using ALoader = uzu::matmul::ThreadgroupLoader<
        int8_t,
        BLOCK_M,
        BLOCK_K,
        Core::SHARED_STRIDE_A,
        true,
        Core::THREADGROUP_THREADS,
        8,
        A_READS,
        A_COLS,
        Core::THREADGROUP_THREADS / A_COLS,
        half>;
    const uint block_row = group_y * BLOCK_M;
    const uint block_col = group_x * BLOCK_N;
    const ushort tile_block_rows = ushort(min(uint(BLOCK_M), params->M - block_row));
    const ushort tile_block_cols = ushort(min(uint(BLOCK_N), params->N - block_col));
    const ushort leftover_block_depth = 0;
    threadgroup half* a_shared = stage;
    threadgroup half* b_shared = stage + BLOCK_M * Core::SHARED_STRIDE_A;
    ALoader a_loader(
        activations + size_t(block_row) * params->leading_dimension_a + k_offset,
        params->leading_dimension_a,
        a_shared,
        thread_context
    );
    // One decode pass covers the tile: every thread owns a 16-column window of one weight row
    constexpr ushort B_WINDOWS = BLOCK_K / trellis::LANE_COLUMNS;
    static_assert(Core::THREADGROUP_THREADS / B_WINDOWS >= BLOCK_N, "one decode pass covers the tile");
    const ushort thread_index = simdgroup * METAL_SIMD_SIZE + thread_context.simd_lane_id;
    const uint b_row = thread_index / B_WINDOWS;
    const uint b_window = thread_index % B_WINDOWS * trellis::LANE_COLUMNS;
    TrellisWeightLoader b_loader{
        trellis_format,
        codes,
        (block_col + b_row) * code_row_bytes,
        params->K,
        k_offset + b_window,
        b_row,
        BLOCK_N,
        b_shared + b_row * Core::SHARED_STRIDE_B + b_window,
        BLOCK_K
    };
    typename Core::TileAccumulator slab(thread_context);
    const uint iterations = params->aligned_inner_iterations * (trellis::K_STEP / BLOCK_K);
    dispatch_gemm_alignment(alignment.raw_value | uint(GemmAlignment::K), [&](auto gemm_alignment_mask) {
      for (uint done = 0; done < iterations; done += SLAB_ITERATIONS) {
        slab.c_fragment.clear();
        Core::template k_loop<gemm_alignment_mask.value>(
            a_shared,
            b_shared,
            int(min(SLAB_ITERATIONS, iterations - done)),
            a_loader,
            b_loader,
            slab,
            tile_block_rows,
            tile_block_cols,
            leftover_block_depth
        );
        ushort index = 0;
        slab.for_each_output([&](ushort, ushort, ushort, thread float& sum) {
          products.elements()[index++] += int(sum);
        });
      }
    });
    ushort index = 0;
    slab.for_each_output([&](ushort token_offset, ushort row_offset, ushort lane_column, thread float&) {
      store(
          block_row + slab.simdgroup_row_offset + token_offset,
          block_col + slab.simdgroup_col_offset + row_offset + lane_column,
          products.elements()[index++]
      );
    });
  }
}

KERNEL(GemmTrellisReduce)(
    const device int4* partial_sums,
    const device float4* column_group_sums,
    const device float* activation_scales,
    const device float* row_scales,
    const constant TrellisCodebook& codebook,
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
  const TrellisEpilogue epilogue{codebook, column_group_sums, activation_scales, row_scales, output, output_stride};
  for (uint index = 0u; index < REDUCE_VECTOR_WIDTH; ++index) {
    epilogue.store(partial_sum[index], token, first_row + index);
  }
}
