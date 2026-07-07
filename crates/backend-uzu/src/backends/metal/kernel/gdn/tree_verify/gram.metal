#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../common/thread_context.h"
#include "../../generated/trie.h"
#include "../common/gram.h"
#include "../common/heads.h"

using namespace metal;
using namespace uzu::matmul;
using namespace uzu::trie;

#define TREE_GRAM_ROW_TILE 16u
#define TREE_GRAM_COL_TILE 32u
#define TREE_GRAM_THREADS METAL_SIMD_SIZE
#define TREE_GRAM_INVALID_ROW 0xffffffffu

METAL_FUNC void invert_tree_gram_diagonal_block(
    device float* ainv,
    threadgroup const float* diag_a_tile,
    const uint mat_base,
    const uint row_base,
    const uint tree_size,
    const uint block_size,
    const uint thread_idx
);

// Build tree gram matrices for one (batch, value-head) tile.
//
// q, k:   [B, T, Hg, K], with key head hk = hv / (HV / Hg)
// trie:   [B, T] DFS intervals; col is an ancestor of row when row is in col's interval
// prefix: [B, T, HV] path log-decay prefix from BuildPrefixBeta
// beta:   [B, T, HV] sigmoid gate from BuildPrefixBeta
//
// a_mat[row, col] = beta[row] * exp(prefix[row] - prefix[col]) * dot(k[row], k[col])
//                   for proper ancestors only
// qkd[row, col]   = scale * exp(prefix[row] - prefix[col]) * dot(q[row], k[col])
//                   for ancestor-or-self
// ainv            = (I + A)^-1 for each diagonal block
template <typename T, bool USE_MXU>
VARIANTS(T, float, bfloat)
VARIANTS(USE_MXU, false, true)
PUBLIC KERNEL(BuildTreeGram)(
    const device T* q,
    const device T* k,
    const device TrieNode* trie,
    const device float* prefix,
    const device float* beta,
    device float* a_mat,
    device float* qkd,
    device float* ainv,
    constant const float& scale,
    constant const uint& batch_size,
    constant const uint& tree_size,
    constant const uint& k_heads,
    constant const uint& value_heads,
    constant const uint& head_k_dim,
    threadgroup float diag_a_tile[TREE_GRAM_ROW_TILE * TREE_GRAM_ROW_TILE],
    threadgroup float row_prefix[TREE_GRAM_ROW_TILE],
    threadgroup float row_beta[TREE_GRAM_ROW_TILE],
    threadgroup uint row_token[TREE_GRAM_ROW_TILE],
    threadgroup float col_prefix[TREE_GRAM_COL_TILE],
    threadgroup uint col_trie_start[TREE_GRAM_COL_TILE],
    threadgroup uint col_trie_end[TREE_GRAM_COL_TILE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint value_head_idx GROUPS(value_heads),
    const uint tile_idx GROUPS(tree_size.div_ceil(TREE_GRAM_ROW_TILE) * tree_size.div_ceil(TREE_GRAM_COL_TILE)),
    const uint thread_idx THREADS(TREE_GRAM_THREADS)
) {
  using Ops = metal::conditional_t<USE_MXU, MxuFragmentOps<>, SimdgroupFragmentOps>;
  constexpr ushort ROW_FRAGMENTS = TREE_GRAM_ROW_TILE / Ops::FRAGMENT_ROWS;
  constexpr ushort COL_FRAGMENTS = TREE_GRAM_COL_TILE / Ops::FRAGMENT_COLS;
  using InputType = metal::conditional_t<USE_MXU, T, float>;
  using AccFragment = Fragment<float, ROW_FRAGMENTS, COL_FRAGMENTS, Ops>;
  using LeftFragment = OperandFragment<InputType, ROW_FRAGMENTS, 1, Ops>;
  using RightFragment = OperandFragment<InputType, 1, COL_FRAGMENTS, Ops, ReadTranspose>;

  const uint key_head_idx = gdn_key_head_for_value_head(value_head_idx, value_heads, k_heads);
  const uint qk_stride = k_heads * head_k_dim;
  const uint qk_base = (batch_idx * tree_size * k_heads + key_head_idx) * head_k_dim;
  const uint prefix_base = batch_idx * tree_size * value_heads + value_head_idx;
  const uint trie_base = batch_idx * tree_size;
  const uint mat_base = (batch_idx * value_heads + value_head_idx) * tree_size * tree_size;

  const uint col_tiles = div_ceil(tree_size, TREE_GRAM_COL_TILE);
  const uint row_tile_idx = tile_idx / col_tiles;
  const uint col_tile_idx = tile_idx - row_tile_idx * col_tiles;
  const uint row_base = row_tile_idx * TREE_GRAM_ROW_TILE;
  const uint col_base = col_tile_idx * TREE_GRAM_COL_TILE;
  const ushort lane = thread_context.simd_lane_id;

  if (row_base >= tree_size) {
    return;
  }

  const uint tile_rows = min(TREE_GRAM_ROW_TILE, tree_size - row_base);
  const uint tile_cols = min(TREE_GRAM_COL_TILE, tree_size - col_base);
  const uint tile_base = mat_base + row_base * tree_size + col_base;

  if (col_base >= row_base + TREE_GRAM_ROW_TILE) {
    for (uint idx = thread_idx; idx < tile_rows * tile_cols; idx += TREE_GRAM_THREADS) {
      const uint local_row = idx / tile_cols;
      const uint local_col = idx - local_row * tile_cols;
      const uint offset = tile_base + local_row * tree_size + local_col;
      a_mat[offset] = 0.0f;
      qkd[offset] = 0.0f;
    }
    return;
  }

  AccFragment kk_acc;
  AccFragment qk_acc;
  kk_acc.clear();
  qk_acc.clear();

  for (uint k_block_start = 0; k_block_start < head_k_dim; k_block_start += Ops::FRAGMENT_ROWS) {
    const uint k_remaining = head_k_dim - k_block_start;
    const uint k_tile_size = Ops::FRAGMENT_ROWS;
    const uint valid_k_cols = min(k_remaining, k_tile_size);

    const uint qk_row_base = qk_base + row_base * qk_stride + k_block_start;
    const uint qk_col_base = qk_base + col_base * qk_stride + k_block_start;
    const device T* k_rows = k + qk_row_base;
    const device T* q_rows = q + qk_row_base;
    const device T* k_cols = k + qk_col_base;

    const bool full_k_tile = valid_k_cols == k_tile_size;
    gdn_accumulate_dual_gram_tile<AccFragment, LeftFragment, RightFragment>(
        kk_acc,
        qk_acc,
        k_rows,
        q_rows,
        k_cols,
        int(qk_stride),
        ushort(tile_rows),
        ushort(tile_cols),
        ushort(valid_k_cols),
        full_k_tile && row_base + TREE_GRAM_ROW_TILE <= tree_size,
        full_k_tile && col_base + TREE_GRAM_COL_TILE <= tree_size,
        lane
    );
  }

  const bool has_diag = col_base <= row_base && row_base < col_base + TREE_GRAM_COL_TILE;
  const uint diag_col_offset = has_diag ? row_base - col_base : 0;
  const uint diag_size = min(TREE_GRAM_ROW_TILE, tree_size - row_base);

  if (thread_idx < TREE_GRAM_COL_TILE) {
    const uint col_token = col_base + thread_idx;
    if (col_token < tree_size) {
      const TrieNode node = trie[trie_base + col_token];
      col_trie_start[thread_idx] = node.trie_start;
      col_trie_end[thread_idx] = node.trie_end;
      col_prefix[thread_idx] = prefix[prefix_base + col_token * value_heads];
    } else {
      col_trie_start[thread_idx] = 1;
      col_trie_end[thread_idx] = 0;
      col_prefix[thread_idx] = 0.0f;
    }
  }
  if (thread_idx < TREE_GRAM_ROW_TILE) {
    const uint token = row_base + thread_idx;
    if (token < tree_size) {
      row_token[thread_idx] = token;
      row_prefix[thread_idx] = prefix[prefix_base + token * value_heads];
      row_beta[thread_idx] = beta[prefix_base + token * value_heads];
    } else {
      row_token[thread_idx] = TREE_GRAM_INVALID_ROW;
      row_prefix[thread_idx] = 0.0f;
      row_beta[thread_idx] = 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  AccFragment::zip_for_each_coord(
      lane,
      [&](ushort local_row, ushort local_col, thread float& kk, thread float& qk_dot) {
        const uint row_idx = row_token[local_row];
        const uint col_idx = col_base + local_col;
        const bool in_subtree = row_idx >= col_trie_start[local_col] && row_idx <= col_trie_end[local_col];
        const float decay = exp(row_prefix[local_row] - col_prefix[local_col]);
        const float a_value = in_subtree && row_idx != col_idx ? row_beta[local_row] * decay * kk : 0.0f;
        const float qkd_value = in_subtree ? scale * decay * qk_dot : 0.0f;
        if (has_diag && local_row < diag_size && local_col >= diag_col_offset) {
          const uint diag_col = local_col - diag_col_offset;
          if (diag_col < diag_size) {
            diag_a_tile[local_row * TREE_GRAM_ROW_TILE + diag_col] = a_value;
          }
        }
        kk = a_value;
        qk_dot = qkd_value;
      },
      kk_acc,
      qk_acc
  );

  const short2 tile_dims = short2(tile_cols, tile_rows);
  device float* a_tile = a_mat + tile_base;
  device float* qkd_tile = qkd + tile_base;
  if (row_base + TREE_GRAM_ROW_TILE <= tree_size && col_base + TREE_GRAM_COL_TILE <= tree_size) {
    kk_acc.store(lane, a_tile, tree_size);
    qk_acc.store(lane, qkd_tile, tree_size);
  } else {
    kk_acc.store_safe(lane, a_tile, tree_size, tile_dims);
    qk_acc.store_safe(lane, qkd_tile, tree_size, tile_dims);
  }

  if (has_diag) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    invert_tree_gram_diagonal_block(ainv, diag_a_tile, mat_base, row_base, tree_size, diag_size, thread_idx);
  }
}

METAL_FUNC void invert_tree_gram_diagonal_block(
    device float* ainv,
    threadgroup const float* diag_a_tile,
    const uint mat_base,
    const uint row_base,
    const uint tree_size,
    const uint block_size,
    const uint thread_idx
) {
  if (thread_idx >= block_size) {
    return;
  }

  const uint col = thread_idx;
  float inverse_col[TREE_GRAM_ROW_TILE] = {};
  inverse_col[col] = 1.0f;

  METAL_PRAGMA_UNROLL
  for (uint row = 0; row < TREE_GRAM_ROW_TILE; row++) {
    if (row > col && row < block_size) {
      float acc = 0.0f;
      METAL_PRAGMA_UNROLL
      for (uint prev_row = 0; prev_row < TREE_GRAM_ROW_TILE; prev_row++) {
        if (prev_row < row) {
          acc += diag_a_tile[row * TREE_GRAM_ROW_TILE + prev_row] * inverse_col[prev_row];
        }
      }
      inverse_col[row] = -acc;
    }
  }

  METAL_PRAGMA_UNROLL
  for (uint row = 0; row < TREE_GRAM_ROW_TILE; row++) {
    if (row < block_size) {
      const uint ainv_offset = mat_base + (row_base + row) * tree_size + row_base + col;
      ainv[ainv_offset] = inverse_col[row];
    }
  }
}
