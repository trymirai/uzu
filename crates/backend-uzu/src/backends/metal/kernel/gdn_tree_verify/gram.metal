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
#define TREE_GRAM_AINV_THREADS 64
#define TREE_GRAM_FUSED_SIMDGROUPS 4
#define TREE_GRAM_FUSED_THREADS (TREE_GRAM_FUSED_SIMDGROUPS * METAL_SIMD_SIZE)
#define TREE_GRAM_LOWER_SIZE (TREE_GRAM_MAX_T * (TREE_GRAM_MAX_T + 1) / 2)

// The Rust BuildTreeGramMetalKernel wrapper dispatches BuildTreeGramMat then BuildTreeGramAinv.
// This public entry keeps the CPU/Metal DSL kernel lists aligned.
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
    const uint tid THREADS(1)
) {}

template <bool USE_MASK>
METAL_FUNC void invert_tree_gram_matrix(
    threadgroup const float* a_shared,
    threadgroup float* x_shared,
    threadgroup ulong* row_mask_shared,
    const device TrieNode* trie,
    device float* ainv,
    ulong mat_base,
    ulong trie_base,
    uint tree_size,
    uint tid
) {
  for (uint idx = tid; idx < tree_size * tree_size; idx += TREE_GRAM_FUSED_THREADS) {
    const uint i = idx / tree_size;
    const uint j = idx - i * tree_size;
    ainv[mat_base + (ulong)i * (ulong)tree_size + (ulong)j] = i == j ? 1.0f : 0.0f;
  }
  for (uint i = tid; i < tree_size; i += TREE_GRAM_FUSED_THREADS) {
    x_shared[(i * (i + 1)) / 2 + i] = 1.0f;
  }
  if constexpr (USE_MASK) {
    for (uint i = tid; i < tree_size; i += TREE_GRAM_FUSED_THREADS) {
      ulong mask = 0;
      for (uint j = 0; j < tree_size; ++j) {
        const TrieNode node = trie[trie_base + (ulong)j];
        if (i != j && i >= node.trie_start && i <= node.trie_end) {
          mask |= ulong(1) << j;
        }
      }
      row_mask_shared[i] = mask;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr uint GROUP_WIDTH = 8;
  constexpr uint GROUPS = TREE_GRAM_FUSED_THREADS / GROUP_WIDTH;
  const uint group_id = tid / GROUP_WIDTH;
  const uint group_lane = tid & (GROUP_WIDTH - 1);
  for (uint i = 1; i < tree_size; ++i) {
    const uint row_lower_base = (i * (i + 1)) / 2;
    const ulong row_mask = USE_MASK ? row_mask_shared[i] : ~ulong(0);
    for (uint j = group_id; j < i; j += GROUPS) {
      if (((row_mask >> j) & 1) != 0) {
        float sum = 0.0f;
        for (uint m = j + group_lane; m < i; m += GROUP_WIDTH) {
          if (((row_mask >> m) & 1) != 0) {
            sum += a_shared[i * TREE_GRAM_MAX_T + m] * x_shared[(m * (m + 1)) / 2 + j];
          }
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        sum += simd_shuffle_xor(sum, 4);
        if (group_lane == 0) {
          const float v = -sum;
          x_shared[row_lower_base + j] = v;
          ainv[mat_base + (ulong)i * (ulong)tree_size + (ulong)j] = v;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// K1: tree gram + masked decay.
//
// Inputs:
//   q, k   : [B, T, Hg, K]     value head hv maps to key head hk = hv / (HV / Hg)
//   trie   : [B, T]            Euler intervals for ancestor tests
//   prefix : [B, T, HV] float  G = path cumsum of log-decay from K0
//   beta   : [B, T, HV] float  sigmoid gate from K0
//
// Outputs:
//   A[i,j,hv]   = (i != j && trie[j] contains i) ? beta[i,hv] * exp(prefix[i,hv] - prefix[j,hv]) * dot(k[i,hk],
//   k[j,hk]) : 0 QKD[i,j,hv] = trie[j] contains i ? exp(prefix[i,hv] - prefix[j,hv]) * scale * dot(q[i,hk], k[j,hk]) :
//   0
template <typename T, bool USE_MXU>
VARIANTS(T, float, half, bfloat)
VARIANTS(USE_MXU, false, true)
PUBLIC KERNEL(BuildTreeGramMat)(
    const device T* q,
    const device T* k,
    const device TrieNode* trie,
    const device float* prefix,
    const device float* beta,
    device float* a_mat,
    device float* qkd,
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
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint hv GROUPS(value_heads),
    const uint tile_idx GROUPS(tree_size.div_ceil(TREE_GRAM_ROW_TILE) * tree_size.div_ceil(TREE_GRAM_COL_TILE)),
    const uint tid THREADS(TREE_GRAM_MAT_THREADS)
) {
  if (tree_size > TREE_GRAM_MAX_T) {
    return;
  }

  using Ops = metal::conditional_t<USE_MXU, MxuFragmentOps, SimdgroupFragmentOps>;
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

  const short2 tile_dims = short2(
      short(min(uint(TREE_GRAM_COL_TILE), tree_size - col_base)),
      short(min(uint(TREE_GRAM_ROW_TILE), tree_size - row_base))
  );
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
}

template <typename T, bool USE_MXU, bool USE_MASK>
VARIANTS(T, float, half, bfloat)
VARIANTS(USE_MXU, false, true)
VARIANTS(USE_MASK, false, true)
PUBLIC KERNEL(BuildTreeGramFused)(
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
    threadgroup float a_shared[TREE_GRAM_MAX_T * TREE_GRAM_MAX_T],
    threadgroup float x_shared[TREE_GRAM_LOWER_SIZE],
    threadgroup ulong row_mask_shared[TREE_GRAM_MAX_T],
    threadgroup float row_prefix_shared[TREE_GRAM_MAX_T],
    threadgroup float row_beta_shared[TREE_GRAM_MAX_T],
    threadgroup float col_prefix_shared[TREE_GRAM_COL_TILE],
    threadgroup uint col_start_shared[TREE_GRAM_COL_TILE],
    threadgroup uint col_end_shared[TREE_GRAM_COL_TILE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint hv GROUPS(value_heads),
    const uint tid THREADS(TREE_GRAM_FUSED_THREADS)
) {
  if (tree_size > TREE_GRAM_MAX_T) {
    return;
  }

  using Ops = metal::conditional_t<USE_MXU, MxuFragmentOps, SimdgroupFragmentOps>;
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

  const uint row_base = thread_context.simdgroup_index * TREE_GRAM_ROW_TILE;
  const ushort lane = thread_context.simd_lane_id;

  for (uint i = tid; i < tree_size; i += TREE_GRAM_FUSED_THREADS) {
    row_prefix_shared[i] = prefix[prefix_base + (ulong)i * (ulong)value_heads];
    row_beta_shared[i] = beta[prefix_base + (ulong)i * (ulong)value_heads];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint col_base = 0; col_base < tree_size; col_base += TREE_GRAM_COL_TILE) {
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

    if (row_base < tree_size) {
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

      const short2 tile_dims = short2(
          short(min(uint(TREE_GRAM_COL_TILE), tree_size - col_base)),
          short(min(uint(TREE_GRAM_ROW_TILE), tree_size - row_base))
      );
      kk_acc.store_safe(
          lane,
          a_mat + mat_base + (ulong)row_base * (ulong)tree_size + (ulong)col_base,
          int(tree_size),
          tile_dims
      );
      kk_acc.store_safe(lane, a_shared + row_base * TREE_GRAM_MAX_T + col_base, TREE_GRAM_MAX_T, tile_dims);
      qk_acc.store_safe(
          lane,
          qkd + mat_base + (ulong)row_base * (ulong)tree_size + (ulong)col_base,
          int(tree_size),
          tile_dims
      );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  invert_tree_gram_matrix<
      USE_MASK>(a_shared, x_shared, row_mask_shared, trie, ainv, mat_base, trie_base, tree_size, tid);
}

// K2: inverse columns of (I + A), split by column block.
template <uint COL_BLOCK, bool USE_MASK>
VARIANTS(COL_BLOCK, 4, 8, 16, 32)
VARIANTS(USE_MASK, false, true)
PUBLIC KERNEL(BuildTreeGramAinv)(
    const device float* a_mat,
    const device TrieNode* trie,
    device float* ainv,
    constant const uint& batch_size,
    constant const uint& tree_size,
    constant const uint& value_heads,
    threadgroup float a_shared[TREE_GRAM_MAX_T * TREE_GRAM_MAX_T],
    threadgroup float x_shared[TREE_GRAM_MAX_T * COL_BLOCK],
    threadgroup ulong row_mask_shared[TREE_GRAM_MAX_T],
    const uint batch_idx GROUPS(batch_size),
    const uint hv GROUPS(value_heads),
    const uint col_block_idx GROUPS(tree_size.div_ceil(COL_BLOCK)),
    const uint tid THREADS(TREE_GRAM_AINV_THREADS)
) {
  if (tree_size > TREE_GRAM_MAX_T) {
    return;
  }

  const ulong mat_base = ((ulong)batch_idx * (ulong)value_heads + (ulong)hv) * (ulong)tree_size * (ulong)tree_size;
  const ulong trie_base = (ulong)batch_idx * (ulong)tree_size;
  const uint col_base = col_block_idx * COL_BLOCK;

  for (uint idx = tid; idx < tree_size * tree_size; idx += TREE_GRAM_AINV_THREADS) {
    const uint i = idx / tree_size;
    const uint j = idx - i * tree_size;
    a_shared[i * TREE_GRAM_MAX_T + j] = a_mat[mat_base + (ulong)i * (ulong)tree_size + (ulong)j];
  }
  for (uint idx = tid; idx < tree_size * COL_BLOCK; idx += TREE_GRAM_AINV_THREADS) {
    const uint i = idx / COL_BLOCK;
    const uint local_j = idx - i * COL_BLOCK;
    const uint j = col_base + local_j;
    const float v = j < tree_size && i == j ? 1.0f : 0.0f;
    x_shared[idx] = v;
    if (j < tree_size) {
      ainv[mat_base + (ulong)i * (ulong)tree_size + (ulong)j] = v;
    }
  }
  if constexpr (USE_MASK) {
    for (uint i = tid; i < tree_size; i += TREE_GRAM_AINV_THREADS) {
      ulong mask = 0;
      for (uint j = 0; j < tree_size; ++j) {
        const TrieNode node = trie[trie_base + (ulong)j];
        if (i != j && i >= node.trie_start && i <= node.trie_end) {
          mask |= ulong(1) << j;
        }
      }
      row_mask_shared[i] = mask;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = 1; i < tree_size; ++i) {
    const ulong row_mask = USE_MASK ? row_mask_shared[i] : ~ulong(0);
    for (uint local_j = tid; local_j < COL_BLOCK; local_j += TREE_GRAM_AINV_THREADS) {
      const uint j = col_base + local_j;
      if (j < i && j < tree_size && ((row_mask >> j) & 1) != 0) {
        float sum = 0.0f;
        for (uint m = j; m < i; ++m) {
          if (((row_mask >> m) & 1) != 0) {
            sum += a_shared[i * TREE_GRAM_MAX_T + m] * x_shared[m * COL_BLOCK + local_j];
          }
        }
        const float v = -sum;
        x_shared[i * COL_BLOCK + local_j] = v;
        ainv[mat_base + (ulong)i * (ulong)tree_size + (ulong)j] = v;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
