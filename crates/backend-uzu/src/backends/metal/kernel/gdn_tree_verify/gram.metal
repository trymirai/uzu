#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/trie.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/mxu_fragment_ops.h"
#include "../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;
using namespace uzu::trie;

#define TREE_GRAM_ROW_TILE 16
#define TREE_GRAM_COL_TILE 32
#define TREE_GRAM_THREADS METAL_SIMD_SIZE

template <typename T, bool USE_MXU>
VARIANTS(T, float, half, bfloat)
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
    threadgroup float diagonal_a_tile[TREE_GRAM_ROW_TILE * TREE_GRAM_ROW_TILE],
    threadgroup float row_prefix_tile[TREE_GRAM_ROW_TILE],
    threadgroup float row_beta_tile[TREE_GRAM_ROW_TILE],
    threadgroup uint row_token_tile[TREE_GRAM_ROW_TILE],
    threadgroup float col_prefix_tile[TREE_GRAM_COL_TILE],
    threadgroup uint col_subtree_start[TREE_GRAM_COL_TILE],
    threadgroup uint col_subtree_end[TREE_GRAM_COL_TILE],
    const ThreadContext thread_context,
    const uint batch GROUPS(batch_size),
    const uint value_head GROUPS(value_heads),
    const uint tile_index GROUPS(tree_size.div_ceil(TREE_GRAM_ROW_TILE) * tree_size.div_ceil(TREE_GRAM_COL_TILE)),
    const uint tid THREADS(TREE_GRAM_THREADS)
) {
  using Ops = metal::conditional_t<USE_MXU, MxuFragmentOps<>, SimdgroupFragmentOps>;
  constexpr ushort ROW_FRAGMENTS = TREE_GRAM_ROW_TILE / Ops::FRAGMENT_ROWS;
  constexpr ushort COL_FRAGMENTS = TREE_GRAM_COL_TILE / Ops::FRAGMENT_COLS;
  using InputType = metal::conditional_t<USE_MXU, T, float>;
  using AccFragment = Fragment<float, ROW_FRAGMENTS, COL_FRAGMENTS, Ops>;
  using LeftFragment = OperandFragment<InputType, ROW_FRAGMENTS, 1, Ops>;
  using RightFragment = OperandFragment<InputType, 1, COL_FRAGMENTS, Ops, ReadTranspose>;

  const uint value_heads_per_key_head = value_heads / k_heads;
  const uint key_head_index = value_head / value_heads_per_key_head;
  const uint qk_token_stride = k_heads * head_k_dim;
  const uint qk_key_head_offset = (batch * tree_size * k_heads + key_head_index) * head_k_dim;
  const uint prefix_value_head_offset = batch * tree_size * value_heads + value_head;
  const uint trie_batch_offset = batch * tree_size;
  const uint matrix_head_offset = (batch * value_heads + value_head) * tree_size * tree_size;

  const uint col_tiles = (tree_size + TREE_GRAM_COL_TILE - 1) / TREE_GRAM_COL_TILE;
  const uint row_tile_index = tile_index / col_tiles;
  const uint col_tile_index = tile_index - row_tile_index * col_tiles;
  const uint tile_row_start = row_tile_index * TREE_GRAM_ROW_TILE;
  const uint tile_col_start = col_tile_index * TREE_GRAM_COL_TILE;
  const ushort lane = thread_context.simd_lane_id;

  AccFragment kk_acc;
  AccFragment qk_acc;
  kk_acc.clear();
  qk_acc.clear();

  for (uint k_block_start = 0; k_block_start < head_k_dim; k_block_start += Ops::FRAGMENT_ROWS) {
    const uint k_remaining = head_k_dim - k_block_start;
    const ushort valid_k_cols = ushort(min(k_remaining, uint(Ops::FRAGMENT_ROWS)));

    if (tile_row_start < tree_size) {
      LeftFragment k_left;
      LeftFragment q_left;
      RightFragment k_right;

      const short row_limit = short(max(int(0), int(tree_size) - int(tile_row_start)));
      const short col_limit = short(max(int(0), int(tree_size) - int(tile_col_start)));

      const uint qk_row_tile_offset = qk_key_head_offset + tile_row_start * qk_token_stride + k_block_start;
      const uint qk_col_tile_offset = qk_key_head_offset + tile_col_start * qk_token_stride + k_block_start;
      const device T* k_rows = k + qk_row_tile_offset;
      const device T* q_rows = q + qk_row_tile_offset;
      const device T* k_cols = k + qk_col_tile_offset;

      const bool full_k_tile = valid_k_cols == Ops::FRAGMENT_ROWS;
      if (full_k_tile && tile_row_start + TREE_GRAM_ROW_TILE <= tree_size) {
        k_left.load_from(lane, fragment_source(k_rows, int(qk_token_stride)));
        q_left.load_from(lane, fragment_source(q_rows, int(qk_token_stride)));
      } else {
        k_left.load_from(lane, fragment_source(k_rows, int(qk_token_stride)).bounded(row_limit, short(valid_k_cols)));
        q_left.load_from(lane, fragment_source(q_rows, int(qk_token_stride)).bounded(row_limit, short(valid_k_cols)));
      }
      if (full_k_tile && tile_col_start + TREE_GRAM_COL_TILE <= tree_size) {
        k_right.load_from(lane, fragment_source(k_cols, int(qk_token_stride)));
      } else {
        k_right.load_from(lane, fragment_source(k_cols, int(qk_token_stride)).bounded(col_limit, short(valid_k_cols)));
      }

      fragment_mma(kk_acc, k_left, k_right);
      fragment_mma(qk_acc, q_left, k_right);
    }
  }

  if (tile_row_start >= tree_size) {
    return;
  }

  const bool has_diagonal_block =
      tile_col_start <= tile_row_start && tile_row_start < tile_col_start + TREE_GRAM_COL_TILE;
  const uint diagonal_col_offset = has_diagonal_block ? tile_row_start - tile_col_start : 0;
  const uint diagonal_block_size = min(uint(TREE_GRAM_ROW_TILE), tree_size - tile_row_start);

  if (tid < TREE_GRAM_COL_TILE) {
    const uint col_token = tile_col_start + tid;
    if (col_token < tree_size) {
      const TrieNode node = trie[trie_batch_offset + col_token];
      col_subtree_start[tid] = node.trie_start;
      col_subtree_end[tid] = node.trie_end;
      col_prefix_tile[tid] = prefix[prefix_value_head_offset + col_token * value_heads];
    } else {
      col_subtree_start[tid] = 1;
      col_subtree_end[tid] = 0;
      col_prefix_tile[tid] = 0.0f;
    }
  }
  if (tid < TREE_GRAM_ROW_TILE) {
    const uint row_token = tile_row_start + tid;
    if (row_token < tree_size) {
      row_token_tile[tid] = row_token;
      row_prefix_tile[tid] = prefix[prefix_value_head_offset + row_token * value_heads];
      row_beta_tile[tid] = beta[prefix_value_head_offset + row_token * value_heads];
    } else {
      row_token_tile[tid] = 0xffffffff;
      row_prefix_tile[tid] = 0.0f;
      row_beta_tile[tid] = 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  kk_acc.zip_map_coords(lane, qk_acc, [&](short row, short col, float kk, float qk_dot) {
    const uint local_row = uint(row);
    const uint local_col = uint(col);
    const uint row_token = row_token_tile[local_row];
    const uint col_token = tile_col_start + local_col;
    const bool col_contains_row = row_token >= col_subtree_start[local_col] && row_token <= col_subtree_end[local_col];
    const float decay = exp(row_prefix_tile[local_row] - col_prefix_tile[local_col]);
    const float a_value = col_contains_row && row_token != col_token ? row_beta_tile[local_row] * decay * kk : 0.0f;
    const float qkd_value = col_contains_row ? scale * decay * qk_dot : 0.0f;
    if (has_diagonal_block && local_row < diagonal_block_size && local_col >= diagonal_col_offset) {
      const uint local_diag_col = local_col - diagonal_col_offset;
      if (local_diag_col < diagonal_block_size) {
        diagonal_a_tile[local_row * TREE_GRAM_ROW_TILE + local_diag_col] = a_value;
      }
    }
    return float2(a_value, qkd_value);
  });

  const uint tile_rows = min(uint(TREE_GRAM_ROW_TILE), tree_size - tile_row_start);
  const uint tile_cols = min(uint(TREE_GRAM_COL_TILE), tree_size - tile_col_start);
  const short2 tile_dims = short2(short(tile_cols), short(tile_rows));
  const uint matrix_tile_offset = matrix_head_offset + tile_row_start * tree_size + tile_col_start;
  device float* a_tile = a_mat + matrix_tile_offset;
  device float* qkd_tile = qkd + matrix_tile_offset;
  if (tile_row_start + TREE_GRAM_ROW_TILE <= tree_size && tile_col_start + TREE_GRAM_COL_TILE <= tree_size) {
    kk_acc.store(lane, a_tile, int(tree_size));
    qk_acc.store(lane, qkd_tile, int(tree_size));
  } else {
    kk_acc.store_safe(lane, a_tile, int(tree_size), tile_dims);
    qk_acc.store_safe(lane, qkd_tile, int(tree_size), tile_dims);
  }

  // Invert each diagonal block by column-wise forward substitution on I + A.
  if (has_diagonal_block) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint block_size = diagonal_block_size;
    const uint j = tid;
    if (j < block_size) {
      float xcol[16] = {};
      xcol[j] = 1.0f;
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < 16; i++) {
        if (i > j && i < block_size) {
          float acc = 0.0f;
          METAL_PRAGMA_UNROLL
          for (uint prev_row = 0; prev_row < 16; prev_row++) {
            if (prev_row < i) {
              acc += diagonal_a_tile[i * TREE_GRAM_ROW_TILE + prev_row] * xcol[prev_row];
            }
          }
          xcol[i] = -acc;
        }
      }
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < 16; i++) {
        if (i < block_size) {
          const uint ainv_offset = matrix_head_offset + (tile_row_start + i) * tree_size + tile_row_start + j;
          ainv[ainv_offset] = xcol[i];
        }
      }
    }
  }
}
