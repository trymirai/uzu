#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;

#define CHUNK_GRAM_ROW_TILE 16u
#define CHUNK_GRAM_COL_TILE 32u

template <uint HEAD_K_DIM, uint CHUNK_SIZE>
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
PUBLIC KERNEL(DeltaNetChunkedGram)(
    device const float* q_norm,
    device const float* k_norm,
    device float* kk_out,
    device float* qk_out,
    constant const uint& num_k_heads,
    constant const uint& key_dim,
    constant const uint& suffix_len,
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hk_idx GROUPS(num_k_heads),
    const uint tile_idx GROUPS(CHUNK_SIZE.div_ceil(CHUNK_GRAM_ROW_TILE) * CHUNK_SIZE.div_ceil(CHUNK_GRAM_COL_TILE)),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  using Ops = SimdgroupFragmentOps;
  constexpr ushort ROW_FRAGMENTS = CHUNK_GRAM_ROW_TILE / Ops::FRAGMENT_ROWS;
  constexpr ushort COL_FRAGMENTS = CHUNK_GRAM_COL_TILE / Ops::FRAGMENT_COLS;
  using AccFragment = Fragment<float, ROW_FRAGMENTS, COL_FRAGMENTS, Ops>;
  using LeftFragment = OperandFragment<float, ROW_FRAGMENTS, 1, Ops>;
  using RightFragment = OperandFragment<float, 1, COL_FRAGMENTS, Ops, ReadTranspose>;

  static_assert(HEAD_K_DIM % Ops::FRAGMENT_ROWS == 0, "HEAD_K_DIM must align to fragment K");

  constexpr uint row_tiles = CHUNK_SIZE / CHUNK_GRAM_ROW_TILE;
  constexpr uint col_tiles = (CHUNK_SIZE + CHUNK_GRAM_COL_TILE - 1) / CHUNK_GRAM_COL_TILE;
  const uint row_tile_idx = tile_idx / col_tiles;
  const uint col_tile_idx = tile_idx - row_tile_idx * col_tiles;
  const uint row_base = row_tile_idx * CHUNK_GRAM_ROW_TILE;
  const uint col_base = col_tile_idx * CHUNK_GRAM_COL_TILE;
  const uint chunk_token_base = chunk_idx * CHUNK_SIZE;

  const uint tile_rows = min(uint(CHUNK_GRAM_ROW_TILE), uint(CHUNK_SIZE - row_base));
  const uint tile_cols = min(uint(CHUNK_GRAM_COL_TILE), uint(CHUNK_SIZE - col_base));
  const uint row_token_base = chunk_token_base + row_base;
  const uint col_token_base = chunk_token_base + col_base;
  const uint valid_rows = row_token_base < suffix_len ? min(tile_rows, suffix_len - row_token_base) : 0u;
  const uint valid_cols = col_token_base < suffix_len ? min(tile_cols, suffix_len - col_token_base) : 0u;

  AccFragment kk_acc;
  AccFragment qk_acc;
  kk_acc.clear();
  qk_acc.clear();

  for (uint k_block_start = 0; k_block_start < HEAD_K_DIM; k_block_start += Ops::FRAGMENT_ROWS) {
    LeftFragment k_left;
    LeftFragment q_left;
    RightFragment k_right;

    const device float* k_rows = k_norm + row_token_base * key_dim + hk_idx * HEAD_K_DIM + k_block_start;
    const device float* q_rows = q_norm + row_token_base * key_dim + hk_idx * HEAD_K_DIM + k_block_start;
    const device float* k_cols = k_norm + col_token_base * key_dim + hk_idx * HEAD_K_DIM + k_block_start;

    if (valid_rows == tile_rows) {
      k_left.load_from(lane, fragment_source(k_rows, key_dim));
      q_left.load_from(lane, fragment_source(q_rows, key_dim));
    } else {
      k_left.load_from(lane, fragment_source(k_rows, key_dim).bounded(valid_rows, Ops::FRAGMENT_ROWS));
      q_left.load_from(lane, fragment_source(q_rows, key_dim).bounded(valid_rows, Ops::FRAGMENT_ROWS));
    }
    if (valid_cols == tile_cols) {
      k_right.load_from(lane, fragment_source(k_cols, key_dim));
    } else {
      k_right.load_from(lane, fragment_source(k_cols, key_dim).bounded(valid_cols, Ops::FRAGMENT_ROWS));
    }

    fragment_mma(kk_acc, k_left, k_right);
    fragment_mma(qk_acc, q_left, k_right);
  }

  const uint out_base = (chunk_idx * num_k_heads + hk_idx) * CHUNK_SIZE * CHUNK_SIZE + row_base * CHUNK_SIZE + col_base;
  const short2 tile_dims = short2(tile_cols, tile_rows);
  kk_acc.store_safe(lane, kk_out + out_base, CHUNK_SIZE, tile_dims);
  qk_acc.store_safe(lane, qk_out + out_base, CHUNK_SIZE, tile_dims);
}

template <uint CHUNK_SIZE>
VARIANTS(CHUNK_SIZE, 16, 32, 64)
PUBLIC KERNEL(DeltaNetChunkedScaleQk)(
    device const float* qk,
    device const float* g,
    device float* qk_scaled,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& suffix_len,
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint tid THREADS(256)
) {
  const uint token_base = chunk_idx * CHUNK_SIZE;
  const uint valid_tokens = token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - token_base) : 0u;
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk_idx = hv_idx / groups_per_head;
  const uint src_base = (chunk_idx * num_k_heads + hk_idx) * CHUNK_SIZE * CHUNK_SIZE;
  const uint dst_base = (chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE * CHUNK_SIZE;

  for (uint index = tid; index < CHUNK_SIZE * CHUNK_SIZE; index += 256) {
    const uint row = index / CHUNK_SIZE;
    const uint col = index - row * CHUNK_SIZE;
    if (row >= valid_tokens || col >= valid_tokens || col > row) {
      qk_scaled[dst_base + index] = 0.0f;
      continue;
    }
    const float g_row = g[(token_base + row) * num_v_heads + hv_idx];
    const float g_col = g[(token_base + col) * num_v_heads + hv_idx];
    qk_scaled[dst_base + index] = qk[src_base + index] * fast::exp(g_row - g_col);
  }
}
