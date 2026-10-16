#pragma once

#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../matmul/common/mxu_fragment/ops.h"

using namespace metal;

#define ATTENTION_GQA_SIMDGROUPS 8

template <uint BK, uint BD, uint D_SLICES>
struct AttentionGemmGroupedLayout {
  using Ops = uzu::matmul::MxuFragmentOps<>;

  UZU_CONST int SIMDGROUPS = ATTENTION_GQA_SIMDGROUPS;
  UZU_CONST int TILE_ROWS = Ops::FRAGMENT_ROWS;
  UZU_CONST int SLICE_COLS = int(BD) / int(D_SLICES);
  UZU_CONST int BANDS = SIMDGROUPS / int(D_SLICES);
  UZU_CONST int BLOCK_ROWS = BANDS * TILE_ROWS;
  UZU_CONST int SCORE_TILE_ELEMENTS = TILE_ROWS * int(BK);
  UZU_CONST int SCORE_EXCHANGE_SIZE = SIMDGROUPS * SCORE_TILE_ELEMENTS;
};
