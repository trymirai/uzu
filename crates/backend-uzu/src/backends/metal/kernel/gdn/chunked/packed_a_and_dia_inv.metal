#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../common/tri_inv.h"

using namespace metal;

#define CAUSAL_BLOCK_SIZE 16u

template <uint CHUNK_SIZE>
VARIANTS(CHUNK_SIZE, 16, 32, 64)
KERNEL(DeltaNetChunkedPackedAAndDiaInv)(
    device const float* kk,
    device const float* beta,
    device const float* g,
    device float* a_packed,
    device float* a_inv,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& suffix_len,
    threadgroup float diag_a_tile[CAUSAL_BLOCK_SIZE * CAUSAL_BLOCK_SIZE],
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint block_idx GROUPS(CHUNK_SIZE.div_ceil(CAUSAL_BLOCK_SIZE)),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  constexpr uint num_blocks = (CHUNK_SIZE + CAUSAL_BLOCK_SIZE - 1) / CAUSAL_BLOCK_SIZE;
  constexpr uint num_col_pairs = (num_blocks + 1) / 2;
  const uint hk_idx = hv_idx / (num_v_heads / num_k_heads);
  const uint token_base = chunk_idx * CHUNK_SIZE;
  const uint row_base = block_idx * CAUSAL_BLOCK_SIZE;
  const uint kk_base = (chunk_idx * num_k_heads + hk_idx) * CHUNK_SIZE * CHUNK_SIZE;
  const uint diag_pair = block_idx / 2;
  const uint diag_col_offset = (block_idx % 2) * CAUSAL_BLOCK_SIZE;
  const device float* g_head = g + hv_idx * suffix_len;

  for (uint idx = lane; idx < num_col_pairs * CAUSAL_BLOCK_SIZE * 2 * CAUSAL_BLOCK_SIZE; idx += METAL_SIMD_SIZE) {
    const uint local_col = idx % (2 * CAUSAL_BLOCK_SIZE);
    const uint local_row = (idx / (2 * CAUSAL_BLOCK_SIZE)) % CAUSAL_BLOCK_SIZE;
    const uint pair_idx = idx / (CAUSAL_BLOCK_SIZE * 2 * CAUSAL_BLOCK_SIZE);
    const uint row = row_base + local_row;
    const uint col = pair_idx * 2 * CAUSAL_BLOCK_SIZE + local_col;
    const uint row_token = token_base + row;
    const uint col_token = token_base + col;

    float value = 0.0f;
    if (row < CHUNK_SIZE && col < CHUNK_SIZE && row_token < suffix_len && col_token < suffix_len && col < row) {
      const float beta_row = beta[row_token * num_v_heads + hv_idx];
      const float g_row = g_head[row_token];
      const float g_col = g_head[col_token];
      value = beta_row * fast::exp(g_row - g_col) * kk[kk_base + row * CHUNK_SIZE + col];
    }

    const uint out_idx = (((chunk_idx * num_v_heads + hv_idx) * num_blocks + block_idx) * num_col_pairs + pair_idx) *
                             (CAUSAL_BLOCK_SIZE * 2 * CAUSAL_BLOCK_SIZE) +
                         local_row * (2 * CAUSAL_BLOCK_SIZE) + local_col;
    a_packed[out_idx] = value;
    if (pair_idx == diag_pair && local_col >= diag_col_offset && local_col < diag_col_offset + CAUSAL_BLOCK_SIZE) {
      diag_a_tile[local_row * CAUSAL_BLOCK_SIZE + local_col - diag_col_offset] = value;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float* a_inv_block =
      a_inv + ((chunk_idx * num_v_heads + hv_idx) * num_blocks + block_idx) * (CAUSAL_BLOCK_SIZE * CAUSAL_BLOCK_SIZE);
  const uint block_tokens =
      token_base + row_base < suffix_len ? min(uint(CAUSAL_BLOCK_SIZE), suffix_len - token_base - row_base) : 0u;
  invert_lower_triangular_block<CAUSAL_BLOCK_SIZE>(a_inv_block, diag_a_tile, block_tokens, lane);
}
