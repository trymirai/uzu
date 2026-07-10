#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../common/gram.h"
#include "../common/tri_inv.h"

using namespace metal;
using namespace uzu::matmul;

#define ROW_TILE 16u
#define COL_TILE 32u
#define DIAG_BLOCK_SIZE 16u

template <uint HEAD_K_DIM, uint CHUNK_SIZE>
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 32, 64)
KERNEL(DeltaNetChunkedGramA)(
    device const float* q_norm,
    device const float* k_norm,
    device const float* g,
    device const float* beta,
    device float* qk_scaled_out,
    device float* a_packed,
    device float* a_inv,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& key_dim,
    constant const uint& suffix_len,
    threadgroup float diag_a_tile[DIAG_BLOCK_SIZE * DIAG_BLOCK_SIZE],
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
  constexpr uint num_blocks = (CHUNK_SIZE + DIAG_BLOCK_SIZE - 1) / DIAG_BLOCK_SIZE;
  constexpr uint num_col_pairs = (num_blocks + 1) / 2;

  const uint row_tile_idx = tile_idx / col_tiles;
  const uint col_tile_idx = tile_idx - row_tile_idx * col_tiles;
  const uint row_base = row_tile_idx * ROW_TILE;
  const uint col_base = col_tile_idx * COL_TILE;
  const uint chunk_token_base = chunk_idx * CHUNK_SIZE;

  const uint diag_col_pair = row_tile_idx / 2;
  if (col_tile_idx > diag_col_pair) {
    return;
  }
  const bool has_diag = col_tile_idx == diag_col_pair;
  const uint diag_col_offset = (row_tile_idx % 2) * DIAG_BLOCK_SIZE;

  const uint tile_rows = min(uint(ROW_TILE), uint(CHUNK_SIZE - row_base));
  const uint tile_cols = min(uint(COL_TILE), uint(CHUNK_SIZE - col_base));
  const uint row_token_base = chunk_token_base + row_base;
  const uint col_token_base = chunk_token_base + col_base;
  const uint valid_rows = row_token_base < suffix_len ? min(tile_rows, suffix_len - row_token_base) : 0u;
  const uint valid_cols = col_token_base < suffix_len ? min(tile_cols, suffix_len - col_token_base) : 0u;
  const uint valid_tokens = chunk_token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - chunk_token_base) : 0u;

  AccFragment kk_acc;
  AccFragment qk_acc;
  kk_acc.clear();
  qk_acc.clear();

  if (valid_rows > 0) {
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
  }

  const short2 tile_dims = short2(tile_cols, tile_rows);
  const uint groups_per_head = num_v_heads / num_k_heads;
  for (uint group = 0; group < groups_per_head; ++group) {
    const uint hv_idx = hk_idx * groups_per_head + group;
    const device float* g_head = g + hv_idx * suffix_len;

    AccFragment a_frag = kk_acc;
    a_frag.map_coords(lane, [&](short r, short c, float value) {
      const uint row = row_base + uint(r);
      const uint col = col_base + uint(c);
      float a_value = 0.0f;
      if (row < valid_tokens && col < valid_tokens && col < row) {
        const float beta_row = beta[(chunk_token_base + row) * num_v_heads + hv_idx];
        const float g_row = g_head[chunk_token_base + row];
        const float g_col = g_head[chunk_token_base + col];
        a_value = beta_row * fast::exp(g_row - g_col) * value;
      }
      if (has_diag && uint(c) >= diag_col_offset) {
        const uint diag_col = uint(c) - diag_col_offset;
        if (diag_col < DIAG_BLOCK_SIZE) {
          diag_a_tile[uint(r) * DIAG_BLOCK_SIZE + diag_col] = a_value;
        }
      }
      return a_value;
    });

    const uint a_packed_base =
        (((chunk_idx * num_v_heads + hv_idx) * num_blocks + row_tile_idx) * num_col_pairs + col_tile_idx) *
        (DIAG_BLOCK_SIZE * 2 * DIAG_BLOCK_SIZE);
    a_frag.store(lane, a_packed + a_packed_base, int(COL_TILE));

    if (has_diag) {
      simdgroup_barrier(mem_flags::mem_threadgroup);
      device float* a_inv_block =
          a_inv + ((chunk_idx * num_v_heads + hv_idx) * num_blocks + row_tile_idx) * (DIAG_BLOCK_SIZE * DIAG_BLOCK_SIZE);
      invert_lower_triangular_block<DIAG_BLOCK_SIZE>(a_inv_block, diag_a_tile, valid_rows, lane);
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_rows > 0) {
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
}
