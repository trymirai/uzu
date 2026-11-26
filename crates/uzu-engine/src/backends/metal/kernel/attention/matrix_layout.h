#pragma once

#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/kv_layout.h"
#include "../matmul/common/mxu_fragment/ops.h"
#include "../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;

#define ATTENTION_GQA_SIMDGROUPS 8

template <uint BK, uint BD, uint HEAD_DIM_SLICES, bool USE_MXU = true, uint NUM_SIMDGROUPS = ATTENTION_GQA_SIMDGROUPS>
struct AttentionMatrixLayout {
  using Ops = metal::conditional_t<USE_MXU, uzu::matmul::MxuFragmentOps<>, uzu::matmul::SimdgroupFragmentOps>;

  UZU_CONST int SIMDGROUPS = NUM_SIMDGROUPS;
  UZU_CONST int TILE_ROWS = Ops::FRAGMENT_ROWS;
  UZU_CONST int SLICE_COLS = int(BD) / int(HEAD_DIM_SLICES);
  UZU_CONST int BANDS = SIMDGROUPS / int(HEAD_DIM_SLICES);
  UZU_CONST int BLOCK_ROWS = BANDS * TILE_ROWS;
  UZU_CONST int SCORE_TILE_ELEMENTS = TILE_ROWS * int(BK);
};

template <uint HEAD_DIM, bool USE_MXU = true, uint NUM_SIMDGROUPS = ATTENTION_GQA_SIMDGROUPS, bool IS_SPARSE = false>
struct AttentionMatrixScratch {
  using Layout = AttentionMatrixLayout<
      uzu::kv_layout::TILE_KEYS,
      HEAD_DIM,
      HEAD_DIM / uzu::kv_layout::SLICE_DIMS,
      true,
      NUM_SIMDGROUPS>;

  // Exchange offsets are in floats.
  UZU_CONST int STATS_OFFSET = Layout::SIMDGROUPS * Layout::SCORE_TILE_ELEMENTS / int(sizeof(float));
  UZU_CONST int SUMS_OFFSET = STATS_OFFSET + Layout::SIMDGROUPS * Layout::TILE_ROWS * 2;
  UZU_CONST int SLOT_FLOATS = SUMS_OFFSET + Layout::SIMDGROUPS * Layout::TILE_ROWS;
  UZU_CONST int SIMD_KEYS = IS_SPARSE || HEAD_DIM == 512 ? 16 : 32;
  UZU_CONST int CHUNK_KEYS = 128;
  UZU_CONST int CHUNK_TABLE_WORDS = CHUNK_KEYS * int(sizeof(int) + 2 * sizeof(float)) / int(sizeof(ulong));
  UZU_CONST int ROW_WINDOWS =
      USE_MXU ? (IS_SPARSE ? uzu::kv_layout::TILE_KEYS * HEAD_DIM / uzu::kv_layout::SLICE_DIMS : Layout::BLOCK_ROWS)
              : CHUNK_TABLE_WORDS;

  UZU_CONST int QUERY_VECTORS_PER_BAND = int(HEAD_DIM) / (2 * uzu::kv_layout::TILE_DIMS) * METAL_SIMD_SIZE;
  UZU_CONST int QUERY_VECTORS = Layout::BANDS * QUERY_VECTORS_PER_BAND;
  UZU_CONST int EXCHANGE_FLOATS = USE_MXU ? 2 * SLOT_FLOATS : NUM_SIMDGROUPS * 8 * SIMD_KEYS;
  UZU_CONST int STAGING_VECTORS = USE_MXU ? QUERY_VECTORS : SIMD_KEYS * (HEAD_DIM + 8) / 8;

  // Also holds the output tile.
  UZU_CONST int OUTPUT_TILE_BYTES =
      Layout::SIMDGROUPS * Layout::TILE_ROWS * int(uzu::kv_layout::SLICE_DIMS) * int(sizeof(bfloat));
  UZU_CONST int TILE_SCRATCH_VECTORS = USE_MXU && OUTPUT_TILE_BYTES > STAGING_VECTORS* int(sizeof(uint4))
                                           ? OUTPUT_TILE_BYTES / int(sizeof(uint4))
                                           : STAGING_VECTORS;

  // Apple GPU threadgroup limit.
  UZU_CONST int MAX_THREADGROUP_BYTES = 32768;
  UZU_CONST int TOTAL_BYTES = TILE_SCRATCH_VECTORS * int(sizeof(uint4)) + EXCHANGE_FLOATS * int(sizeof(float)) +
                              ROW_WINDOWS * int(sizeof(uint2));
  static_assert(TOTAL_BYTES <= MAX_THREADGROUP_BYTES, "threadgroup memory over 32 KB");
};
