#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../common/thread_context.h"
#include "../../generated/trie.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/mxu_fragment_ops.h"
#include "../../matmul/common/simdgroup_fragment_ops.h"
#include "../common/gram.h"
#include "../common/solve.h"

using namespace metal;
using namespace uzu::matmul;
using namespace uzu::trie;

#define ROW_TILE 16u
#define COL_TILE 32u
#define NUM_SIMDGROUPS 2u
#define INVALID_ROW 0xffffffffu

// kh0 = k @ h0[h0_idx[batch]]^T for one row-tile, walking head_v_dim in COL_TILE
// chunks from dv_start by dv_stride. No-op for a zero initial state.
template <typename T, typename Ops, typename AccFragment, typename LeftFragment, typename RightFragment>
METAL_FUNC void tree_kh0(
    const device T* k_rows,
    const device T* h0,
    const device int* h0_idx,
    device float* kh0,
    const uint batch_idx,
    const uint value_head_idx,
    const uint row_base,
    const uint tile_rows,
    const bool full_rows,
    const uint tree_size,
    const uint value_heads,
    const uint head_k_dim,
    const uint head_v_dim,
    const uint qk_stride,
    const uint dv_start,
    const uint dv_stride,
    const ushort lane
) {
  const int h0_slot = h0_idx[batch_idx];
  if (h0_slot < 0) {
    return;
  }
  const device T* h0_head = h0 + (uint(h0_slot) * value_heads + value_head_idx) * head_v_dim * head_k_dim;
  device float* kh0_rows =
      kh0 + (batch_idx * tree_size + row_base) * value_heads * head_v_dim + value_head_idx * head_v_dim;
  const uint kh0_row_stride = value_heads * head_v_dim;

  for (uint dv_base = dv_start; dv_base < head_v_dim; dv_base += dv_stride) {
    const uint tile_dvs = min(uint(COL_TILE), head_v_dim - dv_base);
    AccFragment kh0_acc;
    kh0_acc.clear();

    const device T* h0_tile = h0_head + dv_base * head_k_dim;
    const bool full_dvs = tile_dvs == COL_TILE;

    for (uint kb = 0; kb < head_k_dim; kb += Ops::FRAGMENT_ROWS) {
      LeftFragment k_left;
      RightFragment h0_right;
      k_left.load_maybe_bounded(lane, k_rows + kb, qk_stride, full_rows, tile_rows, Ops::FRAGMENT_ROWS);
      h0_right.load_maybe_bounded(lane, h0_tile + kb, head_k_dim, full_dvs, tile_dvs, Ops::FRAGMENT_ROWS);
      fragment_mma(kh0_acc, k_left, h0_right);
    }

    kh0_acc.store_maybe_bounded(
        lane,
        kh0_rows + dv_base,
        kh0_row_stride,
        full_rows && full_dvs,
        short2(tile_dvs, tile_rows)
    );
  }
}

// Builds the tree-verify gram products for one (batch, value-head, row-tile,
// col-tile-group) strip; each of the threadgroup's simdgroups owns one 16x32
// column tile (above-diagonal tiles are just zero-filled in qkd) and the row's
// simdgroups share the kh0 dv chunks round-robin:
// a_packed: A[row, col] = beta[row] * exp(prefix[row] - prefix[col]) * dot(k[row], k[col])
//           for proper ancestors (trie interval test), packed block-pair tiles
//           [B*HV, NB, ceil(NB/2), 16, 32] f32; only lower-triangle tiles written
// qkd:      scale * exp(prefix[row] - prefix[col]) * dot(q[row], k[col]) for
//           ancestor-or-self, dense [B*HV, T, T] f32
// a_inv:    (I + A)^-1 per diagonal block, compact [B*HV, NB, 16, 16]
// kh0:      k @ h0[h0_idx[batch]]^T, [B, T, HV, head_v_dim] f32; skipped when
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
    device float* kh0 OPTIONAL(use_h0),
    constant const float& scale,
    constant const uint& batch_size,
    constant const uint& tree_size,
    constant const uint& k_heads,
    constant const uint& value_heads,
    constant const uint& head_k_dim,
    constant const uint& head_v_dim,
    threadgroup float diag_a_tile[ROW_TILE * ROW_TILE],
    threadgroup float row_prefix[ROW_TILE],
    threadgroup float row_beta[ROW_TILE],
    threadgroup uint row_token[ROW_TILE],
    threadgroup float col_prefix[NUM_SIMDGROUPS * COL_TILE],
    threadgroup uint col_trie_start[NUM_SIMDGROUPS * COL_TILE],
    threadgroup uint col_trie_end[NUM_SIMDGROUPS * COL_TILE],
    const bool use_h0 SPECIALIZE,
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint value_head_idx GROUPS(value_heads),
    const uint tile_idx GROUPS(
        tree_size.div_ceil(ROW_TILE) * tree_size.div_ceil(COL_TILE).div_ceil(NUM_SIMDGROUPS)
    ),
    const uint thread_idx THREADS(NUM_SIMDGROUPS * METAL_SIMD_SIZE)
) {
  using Ops = metal::conditional_t<USE_MXU, MxuFragmentOps<>, SimdgroupFragmentOps>;
  constexpr ushort ROW_FRAGMENTS = ROW_TILE / Ops::FRAGMENT_ROWS;
  constexpr ushort COL_FRAGMENTS = COL_TILE / Ops::FRAGMENT_COLS;
  using InputType = metal::conditional_t<USE_MXU, T, float>;
  using AccFragment = Fragment<float, ROW_FRAGMENTS, COL_FRAGMENTS, Ops>;
  using LeftFragment = OperandFragment<InputType, ROW_FRAGMENTS, 1, Ops>;
  using RightFragment = OperandFragment<InputType, 1, COL_FRAGMENTS, Ops, ReadTranspose>;
  static_assert(COL_TILE == METAL_SIMD_SIZE, "COL_TILE must match the SIMD width");

  const uint key_head_idx = value_head_idx / (value_heads / k_heads);
  const uint qk_stride = k_heads * head_k_dim;
  const uint qk_base = (batch_idx * tree_size * k_heads + key_head_idx) * head_k_dim;
  const uint prefix_base = batch_idx * tree_size * value_heads + value_head_idx;
  const uint trie_base = batch_idx * tree_size;
  const uint mat_base = (batch_idx * value_heads + value_head_idx) * tree_size * tree_size;

  const uint col_tiles = div_ceil(tree_size, COL_TILE);
  const uint col_tile_groups = div_ceil(col_tiles, NUM_SIMDGROUPS);
  const uint row_tile_idx = tile_idx / col_tile_groups;
  const uint col_tile_group_idx = tile_idx - row_tile_idx * col_tile_groups;
  const uint row_base = row_tile_idx * ROW_TILE;
  const uint tile_rows = min(ROW_TILE, tree_size - row_base);
  const bool full_rows = row_base + ROW_TILE <= tree_size;
  const uint in_band_col_tiles = min(row_base / COL_TILE + 1, col_tiles);
  const uint diag_col_tile = row_base / COL_TILE;
  const uint diag_col_offset = row_base - diag_col_tile * COL_TILE;
  const ushort lane = thread_context.simd_lane_id;
  const uint simdgroup_idx = thread_context.simdgroup_index;
  const uint col_slice = simdgroup_idx * COL_TILE;
  threadgroup float* sg_col_prefix = col_prefix + col_slice;
  threadgroup uint* sg_col_trie_start = col_trie_start + col_slice;
  threadgroup uint* sg_col_trie_end = col_trie_end + col_slice;

  if (thread_idx < ROW_TILE) {
    const uint token = row_base + thread_idx;
    if (token < tree_size) {
      row_token[thread_idx] = token;
      row_prefix[thread_idx] = prefix[prefix_base + token * value_heads];
      row_beta[thread_idx] = beta[prefix_base + token * value_heads];
    } else {
      row_token[thread_idx] = INVALID_ROW;
      row_prefix[thread_idx] = 0.0f;
      row_beta[thread_idx] = 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint num_blocks = div_ceil(tree_size, ROW_TILE);
  const uint num_col_pairs = div_ceil(num_blocks, 2u);

  const device T* k_rows = k + qk_base + row_base * qk_stride;
  const device T* q_rows = q + qk_base + row_base * qk_stride;

  const uint col_tile_idx = col_tile_group_idx * NUM_SIMDGROUPS + simdgroup_idx;
  if (col_tile_idx < in_band_col_tiles) {
    const uint col_base = col_tile_idx * COL_TILE;
    const uint tile_cols = min(COL_TILE, tree_size - col_base);
    const uint tile_base = mat_base + row_base * tree_size + col_base;

    const uint col_token = col_base + lane;
    if (col_token < tree_size) {
      const TrieNode node = trie[trie_base + col_token];
      sg_col_trie_start[lane] = node.trie_start;
      sg_col_trie_end[lane] = node.trie_end;
      sg_col_prefix[lane] = prefix[prefix_base + col_token * value_heads];
    } else {
      sg_col_trie_start[lane] = 1;
      sg_col_trie_end[lane] = 0;
      sg_col_prefix[lane] = 0.0f;
    }

    const device T* k_cols = k + qk_base + col_base * qk_stride;
    const bool full_cols = col_base + COL_TILE <= tree_size;

    AccFragment kk_acc;
    AccFragment qk_acc;
    kk_acc.clear();
    qk_acc.clear();
    for (uint kb = 0; kb < head_k_dim; kb += Ops::FRAGMENT_ROWS) {
      gdn_accumulate_dual_gram_tile<AccFragment, LeftFragment, RightFragment>(
          kk_acc,
          qk_acc,
          k_rows + kb,
          q_rows + kb,
          k_cols + kb,
          int(qk_stride),
          ushort(tile_rows),
          ushort(tile_cols),
          Ops::FRAGMENT_ROWS,
          full_rows,
          full_cols,
          lane
      );
    }

    const bool has_diag = col_tile_idx == diag_col_tile;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    AccFragment::zip_for_each_coord(
        lane,
        [&](ushort local_row, ushort local_col, thread float& kk, thread float& qk_dot) {
          const uint row_idx = row_token[local_row];
          const uint col_idx = col_base + local_col;
          const bool in_subtree = row_idx >= sg_col_trie_start[local_col] && row_idx <= sg_col_trie_end[local_col];
          const float decay = exp(row_prefix[local_row] - sg_col_prefix[local_col]);
          const float a_value = in_subtree && row_idx != col_idx ? row_beta[local_row] * decay * kk : 0.0f;
          const float qkd_value = in_subtree ? scale * decay * qk_dot : 0.0f;
          if (has_diag && local_row < tile_rows && local_col >= diag_col_offset) {
            const uint diag_col = local_col - diag_col_offset;
            if (diag_col < tile_rows) {
              diag_a_tile[local_row * ROW_TILE + diag_col] = a_value;
            }
          }
          kk = a_value;
          qk_dot = qkd_value;
        },
        kk_acc,
        qk_acc
    );

    device float* a_tile =
        a_packed +
        (((batch_idx * value_heads + value_head_idx) * num_blocks + row_tile_idx) * num_col_pairs + col_tile_idx) *
            (ROW_TILE * COL_TILE);
    device float* qkd_tile = qkd + tile_base;
    kk_acc.store_maybe_bounded(lane, a_tile, COL_TILE, full_rows, short2(COL_TILE, tile_rows));
    qk_acc.store_maybe_bounded(lane, qkd_tile, tree_size, full_rows && full_cols, short2(tile_cols, tile_rows));

    if (has_diag) {
      // diag_a_tile is written and read only by this simdgroup.
      simdgroup_barrier(mem_flags::mem_threadgroup);
      device float* a_inv_block =
          a_inv + ((batch_idx * value_heads + value_head_idx) * num_blocks + row_tile_idx) * (ROW_TILE * ROW_TILE);
      gdn_invert_lower_triangular_block<ROW_TILE>(a_inv_block, diag_a_tile, tile_rows, lane);
    }
  } else if (col_tile_idx < col_tiles) {
    // Above-diagonal zero fill; COL_TILE == METAL_SIMD_SIZE, so each
    // lane owns one column and walks rows by pointer increment.
    const uint col_base = col_tile_idx * COL_TILE;
    const uint tile_cols = min(COL_TILE, tree_size - col_base);
    if (lane < tile_cols) {
      device float* qkd_ptr = qkd + mat_base + row_base * tree_size + col_base + lane;
      for (uint row = 0; row < tile_rows; ++row, qkd_ptr += tree_size) {
        *qkd_ptr = 0.0f;
      }
    }
  }

  // kh0 dv chunks are distributed round-robin over this row tile's simdgroups
  // (across its col-tile-group threadgroups); chunking also bounds registers.
  if (use_h0) {
    const uint row_simdgroups = col_tile_groups * NUM_SIMDGROUPS;
    tree_kh0<T, Ops, AccFragment, LeftFragment, RightFragment>(
        k_rows,
        h0,
        h0_idx,
        kh0,
        batch_idx,
        value_head_idx,
        row_base,
        tile_rows,
        full_rows,
        tree_size,
        value_heads,
        head_k_dim,
        head_v_dim,
        qk_stride,
        col_tile_idx * COL_TILE,
        row_simdgroups * COL_TILE,
        lane
    );
  }
}
