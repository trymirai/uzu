#pragma once

#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../matmul/common/mxu_fragment/ops.h"
#include "../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;

#define ATTENTION_GQA_SIMDGROUPS 8

template <uint BK, uint BD, uint HEAD_DIM_SLICES, bool USE_MXU = true>
struct AttentionGemmGroupedLayout {
  using Ops = metal::conditional_t<USE_MXU, uzu::matmul::MxuFragmentOps<>, uzu::matmul::SimdgroupFragmentOps>;

  UZU_CONST int SIMDGROUPS = ATTENTION_GQA_SIMDGROUPS;
  UZU_CONST int TILE_ROWS = Ops::FRAGMENT_ROWS;
  UZU_CONST int SLICE_COLS = int(BD) / int(HEAD_DIM_SLICES);
  UZU_CONST int BANDS = SIMDGROUPS / int(HEAD_DIM_SLICES);
  UZU_CONST int BLOCK_ROWS = BANDS * TILE_ROWS;
  UZU_CONST int SCORE_TILE_ELEMENTS = TILE_ROWS * int(BK);
};
