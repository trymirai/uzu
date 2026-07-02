#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/trie.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/mxu_fragment_ops.h"
#include "../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;
using namespace uzu::trie;

#define TREE_GRAM_ROW_TILE 16u
#define TREE_GRAM_COL_TILE 32u
#define TREE_GRAM_THREADS METAL_SIMD_SIZE
#define TREE_GRAM_INVALID_ROW 0xffffffffu

// Ragged blocks are padded with identity columns so the 16x16 output is fully
// initialized.
METAL_FUNC void invert_tree_gram_diagonal_block(
    device float* a_inv_block,
    threadgroup const float* diag_a_tile,
    const uint block_size,
    const uint thread_idx
) {
  if (thread_idx >= TREE_GRAM_ROW_TILE) {
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
    a_inv_block[row * TREE_GRAM_ROW_TILE + col] = inverse_col[row];
  }
}

// Builds the tree-verify gram products for one (batch, value-head) 16x32 tile:
// a_packed: A[row, col] = beta[row] * exp(prefix[row] - prefix[col]) * dot(k[row], k[col])
//           for proper ancestors (trie interval test), packed block-pair tiles
//           [B*HV, NB, ceil(NB/2), 16, 32] f32; only lower-triangle tiles written
// qkd:      scale * exp(prefix[row] - prefix[col]) * dot(q[row], k[col]) for
//           ancestor-or-self, dense [B*HV, T, T] f32
// a_inv:    (I + A)^-1 per diagonal block, compact [B*HV, NB, 16, 16]
// kh0:      k @ h0[h0_idx[batch]]^T, [B, T, HV, head_v_dim] in T; skipped when
//           h0_idx[batch] < 0
template <typename T, bool USE_MXU>
VARIANTS(T, float, bfloat)
VARIANTS(USE_MXU, false, true)
PUBLIC KERNEL(BuildTreeGram)(
    const device T* q,
    const device T* k,
    const device TrieNode* trie,
    const device float* prefix,
    const device float* beta,
    const device T* h0 OPTIONAL(use_h0),
    const device int* h0_idx OPTIONAL(use_h0),
    device float* a_packed,
    device float* qkd,
    device float* a_inv,
    device T* kh0 OPTIONAL(use_h0),
    constant const float& scale,
    constant const uint& batch_size,
    constant const uint& tree_size,
    constant const uint& k_heads,
    constant const uint& value_heads,
    constant const uint& head_k_dim,
    constant const uint& head_v_dim,
    threadgroup float diag_a_tile[TREE_GRAM_ROW_TILE * TREE_GRAM_ROW_TILE],
    threadgroup float row_prefix[TREE_GRAM_ROW_TILE],
    threadgroup float row_beta[TREE_GRAM_ROW_TILE],
    threadgroup uint row_token[TREE_GRAM_ROW_TILE],
    threadgroup float col_prefix[TREE_GRAM_COL_TILE],
    threadgroup uint col_trie_start[TREE_GRAM_COL_TILE],
    threadgroup uint col_trie_end[TREE_GRAM_COL_TILE],
    const bool use_h0 SPECIALIZE,
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

  const uint value_heads_per_key_head = value_heads / k_heads;
  const uint key_head_idx = value_head_idx / value_heads_per_key_head;
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
      qkd[tile_base + local_row * tree_size + local_col] = 0.0f;
    }
    return;
  }

  AccFragment kk_acc;
  AccFragment qk_acc;
  kk_acc.clear();
  qk_acc.clear();

  for (uint k_block_start = 0; k_block_start < head_k_dim; k_block_start += Ops::FRAGMENT_ROWS) {
    const uint valid_k_cols = min(head_k_dim - k_block_start, uint(Ops::FRAGMENT_ROWS));

    LeftFragment k_left;
    LeftFragment q_left;
    RightFragment k_right;

    const uint qk_row_base = qk_base + row_base * qk_stride + k_block_start;
    const uint qk_col_base = qk_base + col_base * qk_stride + k_block_start;
    const device T* k_rows = k + qk_row_base;
    const device T* q_rows = q + qk_row_base;
    const device T* k_cols = k + qk_col_base;

    const bool full_k_tile = valid_k_cols == Ops::FRAGMENT_ROWS;
    if (full_k_tile && row_base + TREE_GRAM_ROW_TILE <= tree_size) {
      k_left.load_from(lane, fragment_source(k_rows, qk_stride));
      q_left.load_from(lane, fragment_source(q_rows, qk_stride));
    } else {
      k_left.load_from(lane, fragment_source(k_rows, qk_stride).bounded(tile_rows, valid_k_cols));
      q_left.load_from(lane, fragment_source(q_rows, qk_stride).bounded(tile_rows, valid_k_cols));
    }
    if (full_k_tile && col_base + TREE_GRAM_COL_TILE <= tree_size) {
      k_right.load_from(lane, fragment_source(k_cols, qk_stride));
    } else {
      k_right.load_from(lane, fragment_source(k_cols, qk_stride).bounded(tile_cols, valid_k_cols));
    }

    fragment_mma(kk_acc, k_left, k_right);
    fragment_mma(qk_acc, q_left, k_right);
  }

  const bool has_diag = col_base <= row_base && row_base < col_base + TREE_GRAM_COL_TILE;
  const uint diag_col_offset = has_diag ? row_base - col_base : 0;

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
        if (has_diag && local_row < tile_rows && local_col >= diag_col_offset) {
          const uint diag_col = local_col - diag_col_offset;
          if (diag_col < tile_rows) {
            diag_a_tile[local_row * TREE_GRAM_ROW_TILE + diag_col] = a_value;
          }
        }
        kk = a_value;
        qk_dot = qkd_value;
      },
      kk_acc,
      qk_acc
  );

  const uint num_blocks = div_ceil(tree_size, TREE_GRAM_ROW_TILE);
  const uint num_col_pairs = div_ceil(num_blocks, 2u);
  device float* a_tile =
      a_packed +
      (((batch_idx * value_heads + value_head_idx) * num_blocks + row_tile_idx) * num_col_pairs + col_tile_idx) *
          (TREE_GRAM_ROW_TILE * TREE_GRAM_COL_TILE);
  device float* qkd_tile = qkd + tile_base;
  if (row_base + TREE_GRAM_ROW_TILE <= tree_size) {
    kk_acc.store(lane, a_tile, TREE_GRAM_COL_TILE);
  } else {
    kk_acc.store_safe(lane, a_tile, TREE_GRAM_COL_TILE, short2(TREE_GRAM_COL_TILE, tile_rows));
  }
  if (row_base + TREE_GRAM_ROW_TILE <= tree_size && col_base + TREE_GRAM_COL_TILE <= tree_size) {
    qk_acc.store(lane, qkd_tile, tree_size);
  } else {
    qk_acc.store_safe(lane, qkd_tile, tree_size, short2(tile_cols, tile_rows));
  }

  if (has_diag) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* a_inv_block = a_inv + ((batch_idx * value_heads + value_head_idx) * num_blocks + row_tile_idx) *
                                            (TREE_GRAM_ROW_TILE * TREE_GRAM_ROW_TILE);
    invert_tree_gram_diagonal_block(a_inv_block, diag_a_tile, tile_rows, thread_idx);
  }

  // kh0 dv chunks are distributed round-robin over this row's in-band col-tile
  // TGs (above-diagonal TGs returned early); chunking also bounds registers.
  const uint in_band_col_tiles = min(row_base / TREE_GRAM_COL_TILE + 1, col_tiles);
  if (use_h0 && col_tile_idx < in_band_col_tiles) {
    const int h0_slot = h0_idx[batch_idx];
    if (h0_slot >= 0) {
      const device T* h0_head = h0 + (uint(h0_slot) * value_heads + value_head_idx) * head_v_dim * head_k_dim;
      device T* kh0_rows =
          kh0 + (batch_idx * tree_size + row_base) * value_heads * head_v_dim + value_head_idx * head_v_dim;
      const uint kh0_row_stride = value_heads * head_v_dim;
      const bool full_rows = row_base + TREE_GRAM_ROW_TILE <= tree_size;

      for (uint dv_base = col_tile_idx * TREE_GRAM_COL_TILE; dv_base < head_v_dim;
           dv_base += in_band_col_tiles * TREE_GRAM_COL_TILE) {
        const uint tile_dvs = min(uint(TREE_GRAM_COL_TILE), head_v_dim - dv_base);
        AccFragment kh0_acc;
        kh0_acc.clear();

        for (uint k_block_start = 0; k_block_start < head_k_dim; k_block_start += Ops::FRAGMENT_ROWS) {
          const uint valid_k = min(head_k_dim - k_block_start, uint(Ops::FRAGMENT_ROWS));
          LeftFragment k_left;
          RightFragment h0_right;
          const device T* k_rows = k + qk_base + row_base * qk_stride + k_block_start;
          const device T* h0_tile = h0_head + dv_base * head_k_dim + k_block_start;

          if (valid_k == Ops::FRAGMENT_ROWS && full_rows) {
            k_left.load_from(lane, fragment_source(k_rows, qk_stride));
          } else {
            k_left.load_from(lane, fragment_source(k_rows, qk_stride).bounded(tile_rows, valid_k));
          }
          if (valid_k == Ops::FRAGMENT_ROWS && tile_dvs == TREE_GRAM_COL_TILE) {
            h0_right.load_from(lane, fragment_source(h0_tile, head_k_dim));
          } else {
            h0_right.load_from(lane, fragment_source(h0_tile, head_k_dim).bounded(tile_dvs, valid_k));
          }
          fragment_mma(kh0_acc, k_left, h0_right);
        }

        if (full_rows && tile_dvs == TREE_GRAM_COL_TILE) {
          kh0_acc.store(lane, kh0_rows + dv_base, kh0_row_stride);
        } else {
          kh0_acc.store_safe(lane, kh0_rows + dv_base, kh0_row_stride, short2(tile_dvs, tile_rows));
        }
      }
    }
  }
}
