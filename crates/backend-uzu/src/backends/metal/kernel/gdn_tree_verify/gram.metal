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

#define TREE_GRAM_MAX_T 64
#define TREE_GRAM_ROW_TILE 16
#define TREE_GRAM_COL_TILE 32
#define TREE_GRAM_MAT_THREADS METAL_SIMD_SIZE

template <bool USE_MXU, bool RELAXED_MXU>
// Multiplies a <=16x16 Neumann tile. MXU needs a 16x32 N tile, so only that
// path pads the RHS/result through scratch before copying the valid 16x16 part.
METAL_FUNC void tree_gram_neumann_matmul16(
    threadgroup float* lhs_tile,
    int lhs_stride,
    threadgroup float* rhs_tile,
    int rhs_stride,
    threadgroup float* dst_tile,
    int dst_stride,
    threadgroup float* mxu_rhs_tile_16x32,
    threadgroup float* mxu_dst_tile_16x32,
    ushort lane,
    short valid_rows,
    short valid_cols
) {
  if constexpr (USE_MXU) {
    using Ops = MxuFragmentOps<RELAXED_MXU>;
    using AccFragment = Fragment<float, 1, 2, Ops>;
    using LeftFragment = OperandFragment<float, 1, 1, Ops>;
    using RightFragment = OperandFragment<float, 1, 2, Ops>;

    for (uint idx = lane; idx < 16 * 32; idx += METAL_SIMD_SIZE) {
      const uint r = idx / 32;
      const uint c = idx - r * 32;
      mxu_rhs_tile_16x32[idx] = c < 16 ? rhs_tile[r * rhs_stride + c] : 0.0f;
      mxu_dst_tile_16x32[idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    AccFragment acc;
    LeftFragment left_frag;
    RightFragment right_frag;
    left_frag.load_from(lane, fragment_source(lhs_tile, lhs_stride).bounded(valid_rows, short(16)));
    right_frag.load_from(lane, fragment_source(mxu_rhs_tile_16x32, 32).bounded(short(16), short(32)));
    fragment_mm(acc, left_frag, right_frag);
    acc.store(lane, mxu_dst_tile_16x32, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = lane; idx < 16 * 16; idx += METAL_SIMD_SIZE) {
      const uint r = idx / 16;
      const uint c = idx - r * 16;
      if (r < uint(valid_rows) && c < uint(valid_cols)) {
        dst_tile[r * dst_stride + c] = mxu_dst_tile_16x32[r * 32 + c];
      }
    }
  } else {
    using Ops = SimdgroupFragmentOps;
    using AccFragment = Fragment<float, 2, 2, Ops>;
    using LeftFragment = OperandFragment<float, 2, 2, Ops>;
    using RightFragment = OperandFragment<float, 2, 2, Ops>;
    AccFragment acc;
    LeftFragment left_frag;
    RightFragment right_frag;
    acc.clear();
    left_frag.load_from(lane, fragment_source(lhs_tile, lhs_stride).bounded(valid_rows, short(16)));
    right_frag.load_from(lane, fragment_source(rhs_tile, rhs_stride).bounded(short(16), valid_cols));
    fragment_mma(acc, left_frag, right_frag);
    acc.store_safe(lane, dst_tile, dst_stride, short2(valid_cols, valid_rows));
  }
}

template <bool USE_MXU>
METAL_FUNC void invert_tree_gram_diag_neumann_tile(
    threadgroup float* neumann_power_tile,
    threadgroup float* inverse_tile,
    threadgroup float* product_tile,
    threadgroup float* mxu_scratch,
    device float* ainv,
    ulong mat_base,
    uint tree_size,
    uint block_start,
    uint block_size,
    uint tid
) {
  for (uint covered_power = 1; covered_power < block_size - 1; covered_power = covered_power * 2 + 1) {
    if (tid < METAL_SIMD_SIZE) {
      // Strict MXU precision is intentional here; relaxed precision fails the
      // repeated-squaring Ainv accuracy check.
      tree_gram_neumann_matmul16<USE_MXU, false>(
          neumann_power_tile,
          16,
          neumann_power_tile,
          16,
          product_tile,
          16,
          mxu_scratch,
          mxu_scratch + 16 * 32,
          ushort(tid),
          short(block_size),
          short(block_size)
      );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < 16 * 16; idx += TREE_GRAM_MAT_THREADS) {
      const uint r = idx / 16;
      const uint c = idx - r * 16;
      neumann_power_tile[r * 16 + c] = product_tile[r * 16 + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < METAL_SIMD_SIZE) {
      tree_gram_neumann_matmul16<USE_MXU, false>(
          inverse_tile,
          16,
          neumann_power_tile,
          16,
          product_tile,
          16,
          mxu_scratch,
          mxu_scratch + 16 * 32,
          ushort(tid),
          short(block_size),
          short(block_size)
      );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < block_size * block_size; idx += TREE_GRAM_MAT_THREADS) {
      const uint r = idx / block_size;
      const uint c = idx - r * block_size;
      const uint i = block_start + r;
      const uint j = block_start + c;
      inverse_tile[r * 16 + c] += product_tile[r * 16 + c];
      ainv[mat_base + (ulong)i * (ulong)tree_size + (ulong)j] = inverse_tile[r * 16 + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

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
    threadgroup float row_prefix_shared[TREE_GRAM_MAX_T],
    threadgroup float row_beta_shared[TREE_GRAM_MAX_T],
    threadgroup float col_prefix_shared[TREE_GRAM_COL_TILE],
    threadgroup uint col_start_shared[TREE_GRAM_COL_TILE],
    threadgroup uint col_end_shared[TREE_GRAM_COL_TILE],
    threadgroup float a_tile_shared[TREE_GRAM_ROW_TILE * TREE_GRAM_COL_TILE],
    threadgroup float neumann_power_shared[16 * 16],
    threadgroup float neumann_inverse_shared[16 * 16],
    threadgroup float neumann_product_shared[16 * 16],
    threadgroup float mxu_neumann_scratch[16 * 32 * 2],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint hv GROUPS(value_heads),
    const uint tile_idx GROUPS(tree_size.div_ceil(TREE_GRAM_ROW_TILE) * tree_size.div_ceil(TREE_GRAM_COL_TILE)),
    const uint tid THREADS(TREE_GRAM_MAT_THREADS)
) {
  if (tree_size > TREE_GRAM_MAX_T) {
    return;
  }

  using Ops = metal::conditional_t<USE_MXU, MxuFragmentOps<>, SimdgroupFragmentOps>;
  constexpr ushort ROW_FRAGMENTS = TREE_GRAM_ROW_TILE / Ops::FRAGMENT_ROWS;
  constexpr ushort COL_FRAGMENTS = TREE_GRAM_COL_TILE / Ops::FRAGMENT_COLS;
  using InputType = metal::conditional_t<USE_MXU, T, float>;
  using AccFragment = Fragment<float, ROW_FRAGMENTS, COL_FRAGMENTS, Ops>;
  using LeftFragment = OperandFragment<InputType, ROW_FRAGMENTS, 1, Ops>;
  using RightFragment = OperandFragment<InputType, 1, COL_FRAGMENTS, Ops, ReadTranspose>;

  const uint groups_per_head = value_heads / k_heads;
  const uint hk = hv / groups_per_head;
  const ulong qk_head_base = (((ulong)batch_idx * (ulong)tree_size) * (ulong)k_heads + (ulong)hk) * (ulong)head_k_dim;
  const uint qk_row_stride = k_heads * head_k_dim;
  const ulong prefix_base = ((ulong)batch_idx * (ulong)tree_size) * (ulong)value_heads + (ulong)hv;
  const ulong trie_base = (ulong)batch_idx * (ulong)tree_size;
  const ulong mat_base = ((ulong)batch_idx * (ulong)value_heads + (ulong)hv) * (ulong)tree_size * (ulong)tree_size;

  const uint col_tiles = (tree_size + TREE_GRAM_COL_TILE - 1) / TREE_GRAM_COL_TILE;
  const uint row_tile_idx = tile_idx / col_tiles;
  const uint col_tile_idx = tile_idx - row_tile_idx * col_tiles;
  const uint row_base = row_tile_idx * TREE_GRAM_ROW_TILE;
  const uint col_base = col_tile_idx * TREE_GRAM_COL_TILE;
  const ushort lane = thread_context.simd_lane_id;

  for (uint i = tid; i < tree_size; i += TREE_GRAM_MAT_THREADS) {
    row_prefix_shared[i] = prefix[prefix_base + (ulong)i * (ulong)value_heads];
    row_beta_shared[i] = beta[prefix_base + (ulong)i * (ulong)value_heads];
  }
  if (tid < TREE_GRAM_COL_TILE) {
    const uint j = col_base + tid;
    if (j < tree_size) {
      const TrieNode node = trie[trie_base + (ulong)j];
      col_prefix_shared[tid] = prefix[prefix_base + (ulong)j * (ulong)value_heads];
      col_start_shared[tid] = node.trie_start;
      col_end_shared[tid] = node.trie_end;
    } else {
      col_prefix_shared[tid] = 0.0f;
      col_start_shared[tid] = 1;
      col_end_shared[tid] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

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

  kk_acc.map_coords(lane, [&](short row, short col, float kk) {
    const uint i = row_base + uint(row);
    const uint j = col_base + uint(col);
    if (i >= tree_size || j >= tree_size) {
      return 0.0f;
    }
    const bool incl = i >= col_start_shared[col] && i <= col_end_shared[col];
    const float prefix_i = row_prefix_shared[i];
    const float prefix_j = col_prefix_shared[col];
    const float beta_i = row_beta_shared[i];
    if (!incl || i == j) {
      return 0.0f;
    }
    return beta_i * exp(prefix_i - prefix_j) * kk;
  });

  qk_acc.map_coords(lane, [&](short row, short col, float qk_dot) {
    const uint i = row_base + uint(row);
    const uint j = col_base + uint(col);
    if (i >= tree_size || j >= tree_size) {
      return 0.0f;
    }
    const bool incl = i >= col_start_shared[col] && i <= col_end_shared[col];
    const float prefix_i = row_prefix_shared[i];
    const float prefix_j = col_prefix_shared[col];
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
  kk_acc.store_safe(lane, a_tile_shared, TREE_GRAM_COL_TILE, tile_dims);
  qk_acc.store_safe(
      lane,
      qkd + mat_base + (ulong)row_base * (ulong)tree_size + (ulong)col_base,
      int(tree_size),
      tile_dims
  );

  for (uint idx = tid; idx < tile_rows * tile_cols; idx += TREE_GRAM_MAT_THREADS) {
    const uint i = row_base + idx / tile_cols;
    const uint j = col_base + idx % tile_cols;
    ainv[mat_base + (ulong)i * (ulong)tree_size + (ulong)j] = i == j ? 1.0f : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (col_base <= row_base && row_base < col_base + TREE_GRAM_COL_TILE) {
    const uint diag_col = row_base - col_base;
    const uint block_size = min(uint(16), tree_size - row_base);
    for (uint idx = tid; idx < 16 * 16; idx += TREE_GRAM_MAT_THREADS) {
      const uint r = idx / 16;
      const uint c = idx - r * 16;
      const uint i = row_base + r;
      const uint j = row_base + c;
      const bool in_block = r < block_size && c < block_size;
      const float neg_a = in_block && c < r ? -a_tile_shared[r * TREE_GRAM_COL_TILE + diag_col + c] : 0.0f;
      neumann_power_shared[idx] = neg_a;
      neumann_inverse_shared[idx] = in_block && r == c ? 1.0f : neg_a;
      if (in_block) {
        ainv[mat_base + (ulong)i * (ulong)tree_size + (ulong)j] = neumann_inverse_shared[idx];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    invert_tree_gram_diag_neumann_tile<USE_MXU>(
        neumann_power_shared,
        neumann_inverse_shared,
        neumann_product_shared,
        mxu_neumann_scratch,
        ainv,
        mat_base,
        tree_size,
        row_base,
        block_size,
        tid
    );
  }
}
