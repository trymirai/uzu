#pragma once

#include "../../../common/defines.h"
#include "../../../generated/matmul.h"
#include "../../common/mxu_fragment/ops.h"

namespace uzu {
namespace gemm {

template <uint BITS>
struct MxuTiledLayout {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  UZU_CONST uint BLOCK_ROWS = BITS == 2 ? uzu::matmul::MXU_TILED_W2_BLOCK_ROWS : uzu::matmul::MXU_TILED_BLOCK_ROWS;
  UZU_CONST uint LANE_VECTOR_CHUNKS = uzu::matmul::MXU_TILED_BLOCK_ROWS / BLOCK_ROWS;
  UZU_CONST uint ROW_FRAGMENTS = BLOCK_ROWS / Ops::FRAGMENT_ROWS;
  UZU_CONST uint COL_FRAGMENTS = BITS == 4 ? 1 : 2;
  UZU_CONST uint LANE_WORDS = ROW_FRAGMENTS * Ops::THREAD_ELEMENT_ROWS * COL_FRAGMENTS;
  UZU_CONST uint LANE_BYTES = LANE_WORDS * sizeof(uint);
  UZU_CONST uint BLOCK_CHUNK_BYTES = METAL_SIMD_SIZE * LANE_BYTES;
};

} // namespace gemm
} // namespace uzu
