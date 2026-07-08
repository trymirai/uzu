#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../common/solve.h"

using namespace metal;

#define CHUNK_SOLVE_BLOCK 16u

template <bool RECOMPUTE_G>
METAL_FUNC float chunked_g(
    device const float* g_or_log_decay,
    uint token_base,
    uint local_t,
    uint num_v_heads,
    uint hv_idx
) {
  if constexpr (RECOMPUTE_G) {
    float acc = 0.0f;
    for (uint i = 0; i <= local_t; ++i) {
      acc += g_or_log_decay[(token_base + i) * num_v_heads + hv_idx];
    }
    return acc;
  } else {
    return g_or_log_decay[(token_base + local_t) * num_v_heads + hv_idx];
  }
}

template <uint CHUNK_SIZE, bool RECOMPUTE_G>
VARIANTS(CHUNK_SIZE, 16, 32, 64)
VARIANTS(RECOMPUTE_G, false, true)
KERNEL(DeltaNetChunkedSolve)(
    device const float* kk,
    device const float* beta,
    device const float* g_or_log_decay,
    device float* a_packed,
    device float* a_inv,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& suffix_len,
    threadgroup float diag_a_tile[CHUNK_SOLVE_BLOCK * CHUNK_SOLVE_BLOCK],
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint block_idx GROUPS(CHUNK_SIZE.div_ceil(CHUNK_SOLVE_BLOCK)),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  constexpr uint num_blocks = (CHUNK_SIZE + CHUNK_SOLVE_BLOCK - 1) / CHUNK_SOLVE_BLOCK;
  constexpr uint num_col_pairs = (num_blocks + 1) / 2;
  const uint hk_idx = hv_idx / (num_v_heads / num_k_heads);
  const uint token_base = chunk_idx * CHUNK_SIZE;
  const uint row_base = block_idx * CHUNK_SOLVE_BLOCK;
  const uint kk_base = (chunk_idx * num_k_heads + hk_idx) * CHUNK_SIZE * CHUNK_SIZE;
  const uint diag_pair = block_idx / 2;
  const uint diag_col_offset = (block_idx % 2) * CHUNK_SOLVE_BLOCK;

  for (uint idx = lane; idx < num_col_pairs * CHUNK_SOLVE_BLOCK * 2 * CHUNK_SOLVE_BLOCK; idx += METAL_SIMD_SIZE) {
    const uint local_col = idx % (2 * CHUNK_SOLVE_BLOCK);
    const uint local_row = (idx / (2 * CHUNK_SOLVE_BLOCK)) % CHUNK_SOLVE_BLOCK;
    const uint pair_idx = idx / (CHUNK_SOLVE_BLOCK * 2 * CHUNK_SOLVE_BLOCK);
    const uint row = row_base + local_row;
    const uint col = pair_idx * 2 * CHUNK_SOLVE_BLOCK + local_col;
    const uint row_token = token_base + row;
    const uint col_token = token_base + col;

    float value = 0.0f;
    if (row < CHUNK_SIZE && col < CHUNK_SIZE && row_token < suffix_len && col_token < suffix_len && col < row) {
      const float beta_row = beta[row_token * num_v_heads + hv_idx];
      const float g_row = chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, row, num_v_heads, hv_idx);
      const float g_col = chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, col, num_v_heads, hv_idx);
      value = beta_row * fast::exp(g_row - g_col) * kk[kk_base + row * CHUNK_SIZE + col];
    }

    const uint out_idx = (((chunk_idx * num_v_heads + hv_idx) * num_blocks + block_idx) * num_col_pairs + pair_idx) *
                             (CHUNK_SOLVE_BLOCK * 2 * CHUNK_SOLVE_BLOCK) +
                         local_row * (2 * CHUNK_SOLVE_BLOCK) + local_col;
    a_packed[out_idx] = value;
    if (pair_idx == diag_pair && local_col >= diag_col_offset && local_col < diag_col_offset + CHUNK_SOLVE_BLOCK) {
      diag_a_tile[local_row * CHUNK_SOLVE_BLOCK + local_col - diag_col_offset] = value;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float* a_inv_block =
      a_inv + ((chunk_idx * num_v_heads + hv_idx) * num_blocks + block_idx) * (CHUNK_SOLVE_BLOCK * CHUNK_SOLVE_BLOCK);
  const uint block_tokens =
      token_base + row_base < suffix_len ? min(uint(CHUNK_SOLVE_BLOCK), suffix_len - token_base - row_base) : 0u;
  gdn_invert_lower_triangular_block<CHUNK_SOLVE_BLOCK>(a_inv_block, diag_a_tile, block_tokens, lane);
}
