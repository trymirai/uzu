#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/mxu_fragment_ops.h"
#include "../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;

#define CHUNK_GRAM_ROW_TILE 16u
#define CHUNK_GRAM_COL_TILE 32u

// Gram (fused with the former ScaleQk pass): computes the per-k-head kk block
// (consumed by Solve) and, for each of the k-head's GQA v-heads, the causal-
// masked, decay-scaled qk block qk_scaled[row,col] = qk * exp(g_row - g_col)
// (col <= row, else 0). The scale-qk expansion is folded in here so no separate
// dispatch and no intermediate qk buffer are needed.
//
// USE_MXU selects the 16x16 MXU fragment path (bf16 q/k operands, f32
// accumulation) vs the 8x8 simdgroup path (f32 operands). The kk/qk outputs and
// the decay-scale expansion stay f32 either way; only the matmul OPERANDS are
// rounded to bf16 on the MXU path (both output tiles are even-N, so both matmuls
// are MXU-eligible). Config B (M1-M4 proxy) uses USE_MXU=false.
template <uint HEAD_K_DIM, uint CHUNK_SIZE, bool USE_MXU>
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
VARIANTS(USE_MXU, false, true)
PUBLIC KERNEL(DeltaNetChunkedGram)(
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
    const uint tile_idx GROUPS(CHUNK_SIZE.div_ceil(CHUNK_GRAM_ROW_TILE) * CHUNK_SIZE.div_ceil(CHUNK_GRAM_COL_TILE)),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  using Ops = metal::conditional_t<USE_MXU, MxuFragmentOps<>, SimdgroupFragmentOps>;
  using InputType = metal::conditional_t<USE_MXU, bfloat, float>;
  constexpr ushort ROW_FRAGMENTS = CHUNK_GRAM_ROW_TILE / Ops::FRAGMENT_ROWS;
  constexpr ushort COL_FRAGMENTS = CHUNK_GRAM_COL_TILE / Ops::FRAGMENT_COLS;
  using AccFragment = Fragment<float, ROW_FRAGMENTS, COL_FRAGMENTS, Ops>;
  using LeftFragment = OperandFragment<InputType, ROW_FRAGMENTS, 1, Ops>;
  using RightFragment = OperandFragment<InputType, 1, COL_FRAGMENTS, Ops, ReadTranspose>;

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

  const short2 tile_dims = short2(tile_cols, tile_rows);
  const uint kk_base = (chunk_idx * num_k_heads + hk_idx) * CHUNK_SIZE * CHUNK_SIZE + row_base * CHUNK_SIZE + col_base;
  kk_acc.store_safe(lane, kk_out + kk_base, CHUNK_SIZE, tile_dims);

  // Expand qk to each of the k-head's GQA v-heads, applying the causal mask and
  // the per-v-head decay scale exp(g_row - g_col). The full chunk tile region is
  // written (masked/out-of-range entries as 0), so qk_scaled needs no pre-zero.
  const uint valid_tokens = chunk_token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - chunk_token_base) : 0u;
  const uint groups_per_head = num_v_heads / num_k_heads;
  for (uint group = 0; group < groups_per_head; ++group) {
    const uint hv_idx = hk_idx * groups_per_head + group;
    AccFragment scaled = qk_acc;
    scaled.map_coords(lane, [&](short r, short c, float value) {
      const uint row = row_base + uint(r);
      const uint col = col_base + uint(c);
      if (row >= valid_tokens || col >= valid_tokens || col > row) {
        return 0.0f;
      }
      const float g_row = g[(chunk_token_base + row) * num_v_heads + hv_idx];
      const float g_col = g[(chunk_token_base + col) * num_v_heads + hv_idx];
      return value * fast::exp(g_row - g_col);
    });
    const uint dst_base =
        (chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE * CHUNK_SIZE + row_base * CHUNK_SIZE + col_base;
    scaled.store_safe(lane, qk_scaled_out + dst_base, CHUNK_SIZE, tile_dims);
  }
}
