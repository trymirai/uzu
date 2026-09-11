#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../common/gram.h"

using namespace metal;
using namespace uzu::matmul;

#define ROW_TILE 16u
#define COL_TILE 32u

// q_norm/k_norm: [suffix_len, key_dim], g: [num_v_heads, suffix_len].
// kk_out: [chunks, num_k_heads, CHUNK_SIZE, CHUNK_SIZE].
// qk_scaled_out: [chunks, num_v_heads, CHUNK_SIZE, CHUNK_SIZE].
// Computes K K^T per k-head and decay-scaled Q K^T per grouped v-head.
// MXU did not improve e2e vs simdgroup.
template <uint HEAD_K_DIM, uint CHUNK_SIZE>
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 32, 64)
KERNEL(DeltaNetChunkedGram)(
    device const float* q_norm,
    device const float* k_norm,
    device const float* g,
    device float* kk_out,
    device float* qk_scaled_out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& key_dim,
    constant const uint& suffix_len,
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hk_idx GROUPS(num_k_heads),
    const uint tile_idx GROUPS(CHUNK_SIZE.div_ceil(ROW_TILE) * CHUNK_SIZE.div_ceil(COL_TILE)),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  using Ops = SimdgroupFragmentOps;
  using InputType = float;
  constexpr ushort ROW_FRAGMENTS = ROW_TILE / Ops::FRAGMENT_ROWS;
  constexpr ushort COL_FRAGMENTS = COL_TILE / Ops::FRAGMENT_COLS;
  using AccFragment = Fragment<float, ROW_FRAGMENTS, COL_FRAGMENTS, Ops>;
  using LeftFragment = OperandFragment<InputType, ROW_FRAGMENTS, 1, Ops>;
  using RightFragment = OperandFragment<InputType, 1, COL_FRAGMENTS, Ops, ReadTranspose>;

  static_assert(HEAD_K_DIM % Ops::FRAGMENT_ROWS == 0, "HEAD_K_DIM must align to fragment K");

  constexpr uint col_tiles = (CHUNK_SIZE + COL_TILE - 1) / COL_TILE;
  const uint row_tile_idx = tile_idx / col_tiles;
  const uint col_tile_idx = tile_idx - row_tile_idx * col_tiles;
  const uint row_base = row_tile_idx * ROW_TILE;
  const uint col_base = col_tile_idx * COL_TILE;
  const uint chunk_token_base = chunk_idx * CHUNK_SIZE;

  const uint tile_rows = min(uint(ROW_TILE), uint(CHUNK_SIZE - row_base));
  const uint tile_cols = min(uint(COL_TILE), uint(CHUNK_SIZE - col_base));
  const uint row_token_base = chunk_token_base + row_base;
  const uint col_token_base = chunk_token_base + col_base;
  const uint valid_rows = row_token_base < suffix_len ? min(tile_rows, suffix_len - row_token_base) : 0u;
  const uint valid_cols = col_token_base < suffix_len ? min(tile_cols, suffix_len - col_token_base) : 0u;
  // Upper-causal tiles are zero.
  if (valid_rows == 0 || valid_cols == 0 || col_base >= row_base + valid_rows) {
    return;
  }

  AccFragment kk_acc;
  AccFragment qk_acc;
  kk_acc.clear();
  qk_acc.clear();

  for (uint k_block_start = 0; k_block_start < HEAD_K_DIM; k_block_start += Ops::FRAGMENT_ROWS) {
    const device float* k_rows = k_norm + row_token_base * key_dim + hk_idx * HEAD_K_DIM + k_block_start;
    const device float* q_rows = q_norm + row_token_base * key_dim + hk_idx * HEAD_K_DIM + k_block_start;
    const device float* k_cols = k_norm + col_token_base * key_dim + hk_idx * HEAD_K_DIM + k_block_start;

    accumulate_dual_gram_tile<AccFragment, LeftFragment, RightFragment>(
        kk_acc,
        qk_acc,
        k_rows,
        q_rows,
        k_cols,
        int(key_dim),
        ushort(valid_rows),
        ushort(valid_cols),
        Ops::FRAGMENT_ROWS,
        valid_rows == tile_rows,
        valid_cols == tile_cols,
        ushort(lane)
    );
  }

  const short2 tile_dims = short2(tile_cols, tile_rows);
  const uint kk_base = (chunk_idx * num_k_heads + hk_idx) * CHUNK_SIZE * CHUNK_SIZE + row_base * CHUNK_SIZE + col_base;
  kk_acc.store_safe(lane, kk_out + kk_base, CHUNK_SIZE, tile_dims);

  // Store qk for each grouped v-head with decay scale and causal mask.
  const uint valid_tokens = chunk_token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - chunk_token_base) : 0u;
  const uint groups_per_head = num_v_heads / num_k_heads;
  for (uint group = 0; group < groups_per_head; ++group) {
    const uint hv_idx = hk_idx * groups_per_head + group;
    const device float* g_head = g + hv_idx * suffix_len;
    AccFragment scaled = qk_acc;
    scaled.map_coords(lane, [&](short r, short c, float value) {
      const uint row = row_base + uint(r);
      const uint col = col_base + uint(c);
      if (row >= valid_tokens || col >= valid_tokens || col > row) {
        return 0.0f;
      }
      const float g_row = g_head[chunk_token_base + row];
      const float g_col = g_head[chunk_token_base + col];
      return value * fast::exp(g_row - g_col);
    });
    const uint dst_base =
        (chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE * CHUNK_SIZE + row_base * CHUNK_SIZE + col_base;
    scaled.store_safe(lane, qk_scaled_out + dst_base, CHUNK_SIZE, tile_dims);
  }
}
