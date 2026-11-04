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

template <uint HEAD_DIM, bool USE_MXU = true, uint NUM_SIMDGROUPS = ATTENTION_GQA_SIMDGROUPS>
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
  UZU_CONST int ROW_MASKS = USE_MXU ? Layout::BLOCK_ROWS : 1;

  UZU_CONST int QUERY_VECTORS_PER_BAND = int(HEAD_DIM) / (2 * uzu::kv_layout::TILE_DIMS) * METAL_SIMD_SIZE;
  UZU_CONST int QUERY_VECTORS = Layout::BANDS * QUERY_VECTORS_PER_BAND;
  UZU_CONST int EXCHANGE_FLOATS = USE_MXU ? 2 * SLOT_FLOATS : NUM_SIMDGROUPS * 8 * (HEAD_DIM == 512 ? 16 : 32);
  UZU_CONST int STAGING_VECTORS = USE_MXU ? QUERY_VECTORS : (HEAD_DIM == 512 ? 16 : 32) * (HEAD_DIM + 8) / 8;
};
