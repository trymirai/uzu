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
    threadgroup float diagonal_a[TREE_GRAM_ROW_TILE * TREE_GRAM_ROW_TILE],
    const ThreadContext thread_context,
    const uint batch GROUPS(batch_size),
    const uint value_head GROUPS(value_heads),
    const uint tile_idx GROUPS(tree_size.div_ceil(TREE_GRAM_ROW_TILE) * tree_size.div_ceil(TREE_GRAM_COL_TILE)),
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
  const uint key_head = value_head / value_heads_per_key_head;
  const ulong qk_head_base = (((ulong)batch * (ulong)tree_size) * (ulong)k_heads + (ulong)key_head) * (ulong)head_k_dim;
  const uint qk_row_stride = k_heads * head_k_dim;
  const ulong prefix_base = ((ulong)batch * (ulong)tree_size) * (ulong)value_heads + (ulong)value_head;
  const ulong trie_base = (ulong)batch * (ulong)tree_size;
  const ulong mat_base = ((ulong)batch * (ulong)value_heads + (ulong)value_head) * (ulong)tree_size * (ulong)tree_size;

  const uint col_tiles = (tree_size + TREE_GRAM_COL_TILE - 1) / TREE_GRAM_COL_TILE;
  const uint row_tile_idx = tile_idx / col_tiles;
  const uint col_tile_idx = tile_idx - row_tile_idx * col_tiles;
  const uint row_base = row_tile_idx * TREE_GRAM_ROW_TILE;
  const uint col_base = col_tile_idx * TREE_GRAM_COL_TILE;
  const ushort lane = thread_context.simd_lane_id;

  AccFragment kk_acc;
  AccFragment qk_acc;
  kk_acc.clear();
  qk_acc.clear();

  for (uint k_base = 0; k_base < head_k_dim; k_base += Ops::FRAGMENT_ROWS) {
    const uint k_rem = head_k_dim - k_base;
    const ushort valid_k_cols = ushort(min(k_rem, uint(Ops::FRAGMENT_ROWS)));

    if (row_base < tree_size) {
      LeftFragment k_left;
      LeftFragment q_left;
      RightFragment k_right;

      const short row_limit = short(max(int(0), int(tree_size) - int(row_base)));
      const short col_limit = short(max(int(0), int(tree_size) - int(col_base)));

      const device T* k_rows = k + qk_head_base + (ulong)row_base * (ulong)qk_row_stride + (ulong)k_base;
      const device T* q_rows = q + qk_head_base + (ulong)row_base * (ulong)qk_row_stride + (ulong)k_base;
      const device T* k_col_ptr = k + qk_head_base + (ulong)col_base * (ulong)qk_row_stride + (ulong)k_base;

      k_left.load_from(lane, fragment_source(k_rows, int(qk_row_stride)).bounded(row_limit, short(valid_k_cols)));
      q_left.load_from(lane, fragment_source(q_rows, int(qk_row_stride)).bounded(row_limit, short(valid_k_cols)));
      k_right.load_from(lane, fragment_source(k_col_ptr, int(qk_row_stride)).bounded(col_limit, short(valid_k_cols)));

      fragment_mma(kk_acc, k_left, k_right);
      fragment_mma(qk_acc, q_left, k_right);
    }
  }

  if (row_base >= tree_size) {
    return;
  }

  const bool has_diagonal_block = col_base <= row_base && row_base < col_base + TREE_GRAM_COL_TILE;
  const uint diagonal_col_offset = has_diagonal_block ? row_base - col_base : 0;
  const uint diagonal_block_size = min(uint(TREE_GRAM_ROW_TILE), tree_size - row_base);

  kk_acc.map_coords(lane, [&](short row, short col, float kk) {
    const uint i = row_base + uint(row);
    const uint j = col_base + uint(col);
    if (i >= tree_size || j >= tree_size) {
      return 0.0f;
    }
    const TrieNode node = trie[trie_base + (ulong)j];
    const bool incl = i >= node.trie_start && i <= node.trie_end;
    const float prefix_i = prefix[prefix_base + (ulong)i * (ulong)value_heads];
    const float prefix_j = prefix[prefix_base + (ulong)j * (ulong)value_heads];
    const float beta_i = beta[prefix_base + (ulong)i * (ulong)value_heads];
    const float a = incl && i != j ? beta_i * exp(prefix_i - prefix_j) * kk : 0.0f;
    if (has_diagonal_block && uint(row) < diagonal_block_size && uint(col) >= diagonal_col_offset) {
      const uint local_col = uint(col) - diagonal_col_offset;
      if (local_col < diagonal_block_size) {
        diagonal_a[uint(row) * TREE_GRAM_ROW_TILE + local_col] = a;
      }
    }
    return a;
  });

  qk_acc.map_coords(lane, [&](short row, short col, float qk_dot) {
    const uint i = row_base + uint(row);
    const uint j = col_base + uint(col);
    if (i >= tree_size || j >= tree_size) {
      return 0.0f;
    }
    const TrieNode node = trie[trie_base + (ulong)j];
    const bool incl = i >= node.trie_start && i <= node.trie_end;
    const float prefix_i = prefix[prefix_base + (ulong)i * (ulong)value_heads];
    const float prefix_j = prefix[prefix_base + (ulong)j * (ulong)value_heads];
    if (!incl) {
      return 0.0f;
    }
    return scale * exp(prefix_i - prefix_j) * qk_dot;
  });

  const uint tile_rows = min(uint(TREE_GRAM_ROW_TILE), tree_size - row_base);
  const uint tile_cols = min(uint(TREE_GRAM_COL_TILE), tree_size - col_base);
  const short2 tile_dims = short2(short(tile_cols), short(tile_rows));
  kk_acc.store_safe(
      lane,
      a_mat + mat_base + (ulong)row_base * (ulong)tree_size + (ulong)col_base,
      int(tree_size),
      tile_dims
  );
  qk_acc.store_safe(
      lane,
      qkd + mat_base + (ulong)row_base * (ulong)tree_size + (ulong)col_base,
      int(tree_size),
      tile_dims
  );

  for (uint idx = tid; idx < tile_rows * tile_cols; idx += TREE_GRAM_THREADS) {
    const uint i = row_base + idx / tile_cols;
    const uint j = col_base + idx % tile_cols;
    ainv[mat_base + (ulong)i * (ulong)tree_size + (ulong)j] = i == j ? 1.0f : 0.0f;
  }

  // ponytail: only diagonal 16x16 inverse blocks are produced; downstream reads no off-block Ainv.
  if (has_diagonal_block) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint block_size = diagonal_block_size;
    const uint j = tid;
    if (j < block_size) {
      float xcol[16];
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < 16; i++) {
        if (i >= block_size || i < j) {
          xcol[i] = 0.0f;
        } else if (i == j) {
          xcol[i] = 1.0f;
        } else {
          float acc = 0.0f;
          METAL_PRAGMA_UNROLL
          for (uint prev_row = 0; prev_row < 16; prev_row++) {
            if (prev_row < i) {
              acc += diagonal_a[i * TREE_GRAM_ROW_TILE + prev_row] * xcol[prev_row];
            }
          }
          xcol[i] = -acc;
        }
      }
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < 16; i++) {
        if (i < block_size) {
          ainv[mat_base + (ulong)(row_base + i) * (ulong)tree_size + (ulong)(row_base + j)] = xcol[i];
        }
      }
    }
  }
}
