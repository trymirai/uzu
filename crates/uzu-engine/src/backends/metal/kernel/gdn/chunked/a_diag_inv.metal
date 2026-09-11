#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../common/tri_inv.h"

using namespace metal;

#define DIAG_BLOCK_SIZE 16u

// kk:       [chunks, num_k_heads, CHUNK_SIZE, CHUNK_SIZE] K K^T from Gram.
// beta:     [suffix_len, num_v_heads].
// g:        [num_v_heads, suffix_len] cumulative log decay.
// a_packed: [chunks, num_v_heads, num_blocks, ceil(num_blocks/2),
//            DIAG_BLOCK_SIZE, 2*DIAG_BLOCK_SIZE].
// a_inv:    [chunks, num_v_heads, num_blocks, DIAG_BLOCK_SIZE, DIAG_BLOCK_SIZE].
//
// One threadgroup owns one (chunk, v-head, column-pair, row-block). It materializes
// A[row,col] = beta[row] * exp(g[row] - g[col]) * kk[row,col] for col < row,
// writes packed column-pair blocks through the diagonal pair, copies the
// diagonal block into threadgroup memory, then writes (I + A_diag)^-1 to a_inv.
template <uint CHUNK_SIZE>
VARIANTS(CHUNK_SIZE, 32, 64)
KERNEL(DeltaNetChunkedADiagInv)(
    device const float* kk,
    device const float* beta,
    device const float* g,
    device float* a_packed,
    device float* a_inv,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& suffix_len,
    threadgroup float diag_a_tile[DIAG_BLOCK_SIZE * DIAG_BLOCK_SIZE],
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint block_col_pair_idx GROUPS(
        CHUNK_SIZE.div_ceil(DIAG_BLOCK_SIZE) * CHUNK_SIZE.div_ceil(DIAG_BLOCK_SIZE).div_ceil(2)
    ),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  constexpr uint num_blocks = (CHUNK_SIZE + DIAG_BLOCK_SIZE - 1) / DIAG_BLOCK_SIZE;
  constexpr uint num_col_pairs = (num_blocks + 1) / 2;
  const uint block_idx = block_col_pair_idx / num_col_pairs;
  const uint col_pair_idx = block_col_pair_idx - block_idx * num_col_pairs;
  const uint hk_idx = hv_idx / (num_v_heads / num_k_heads);
  const uint token_base = chunk_idx * CHUNK_SIZE;
  const uint row_base = block_idx * DIAG_BLOCK_SIZE;
  const uint kk_base = (chunk_idx * num_k_heads + hk_idx) * CHUNK_SIZE * CHUNK_SIZE;
  const uint diag_col_pair = block_idx / 2;
  const uint diag_col_offset = (block_idx % 2) * DIAG_BLOCK_SIZE;
  const device float* g_head = g + hv_idx * suffix_len;
  if (col_pair_idx > diag_col_pair) {
    return;
  }

  METAL_PRAGMA_UNROLL
  for (uint idx = lane; idx < DIAG_BLOCK_SIZE * 2 * DIAG_BLOCK_SIZE; idx += METAL_SIMD_SIZE) {
    const uint local_col = idx % (2 * DIAG_BLOCK_SIZE);
    const uint local_row = (idx / (2 * DIAG_BLOCK_SIZE)) % DIAG_BLOCK_SIZE;
    const uint row = row_base + local_row;
    const uint col = col_pair_idx * 2 * DIAG_BLOCK_SIZE + local_col;
    const uint row_token = token_base + row;
    const uint col_token = token_base + col;

    float value = 0.0f;
    if (row < CHUNK_SIZE && col < CHUNK_SIZE && row_token < suffix_len && col_token < suffix_len && col < row) {
      const float beta_row = beta[row_token * num_v_heads + hv_idx];
      const float g_row = g_head[row_token];
      const float g_col = g_head[col_token];
      value = beta_row * fast::exp(g_row - g_col) * kk[kk_base + row * CHUNK_SIZE + col];
    }

    const uint out_idx =
        (((chunk_idx * num_v_heads + hv_idx) * num_blocks + block_idx) * num_col_pairs + col_pair_idx) *
            (DIAG_BLOCK_SIZE * 2 * DIAG_BLOCK_SIZE) +
        local_row * (2 * DIAG_BLOCK_SIZE) + local_col;
    a_packed[out_idx] = value;
    if (col_pair_idx == diag_col_pair && local_col >= diag_col_offset &&
        local_col < diag_col_offset + DIAG_BLOCK_SIZE) {
      diag_a_tile[local_row * DIAG_BLOCK_SIZE + local_col - diag_col_offset] = value;
    }
  }

  if (col_pair_idx != diag_col_pair) {
    return;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float* a_inv_block =
      a_inv + ((chunk_idx * num_v_heads + hv_idx) * num_blocks + block_idx) * (DIAG_BLOCK_SIZE * DIAG_BLOCK_SIZE);
  const uint block_tokens =
      token_base + row_base < suffix_len ? min(uint(DIAG_BLOCK_SIZE), suffix_len - token_base - row_base) : 0u;
  invert_lower_triangular_block<DIAG_BLOCK_SIZE>(a_inv_block, diag_a_tile, block_tokens, lane);
}
