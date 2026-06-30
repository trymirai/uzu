#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/mxu_fragment_ops.h"
#include "../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;

// Solves chunked GDN update coefficients:
//   acc = beta * (v - exp(prefix) * k @ h0)
//   acc -= A[cur_block, prev_block] @ U[prev_block]
//   U[cur_block] = Ainv[cur_block] @ acc
//
// Shapes:
//   k      [B, T, num_k_heads, HEAD_K_DIM]
//   v      [B, T, num_v_heads, head_v_dim]
//   prefix [B, T, num_v_heads]
//   beta   [B, T, num_v_heads]
//   A      [B * num_v_heads, T, T]
//   Ainv   [B * num_v_heads, T, T]
//   h0     [pool, num_v_heads, head_v_dim, HEAD_K_DIM]
//   h0_idx [B], negative means zero initial state
//   U      [B * num_v_heads, T, head_v_dim]
//
// Grid: one simdgroup per (batch, value head, BV value-dim tile). Token blocks
// are solved serially inside the simdgroup because each block depends on
// previous U blocks. Within a block, the three update steps are fragment matmuls:
//   [BT,K]  @ [K,BV]  -> [BT,BV]
//   [BT,BT] @ [BT,BV] -> [BT,BV]
//   [BT,BT] @ [BT,BV] -> [BT,BV]
template <typename T, uint HEAD_K_DIM, uint BT, uint BV, bool USE_MXU>
VARIANTS(T, float, bfloat)
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(BT, 16)
VARIANTS(BV, 16, 32, 64)
VARIANTS(USE_MXU, false, true)
CONSTRAINT(!USE_MXU || T != "float")
CONSTRAINT(!USE_MXU || BV == 32)
PUBLIC KERNEL(GdnTreeUpdateSolve)(
    const device T* k,
    const device T* v,
    const device float* prefix,
    const device float* beta,
    const device float* a,
    const device float* a_inv,
    const device T* h0,
    const device int* h0_idx,
    device float* u,
    constant const uint& batch_size,
    constant const uint& tree_size,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    const uint batch_idx GROUPS(batch_size),
    const uint value_head_idx GROUPS(num_v_heads),
    const uint value_tile_idx GROUPS(head_v_dim.div_ceil(BV)),
    const uint lane_idx THREADS(METAL_SIMD_SIZE)
) {
  using FragmentOps = metal::conditional_t<USE_MXU, MxuFragmentOps<>, SimdgroupFragmentOps>;
  constexpr ushort TOKEN_FRAGMENTS = BT / FragmentOps::FRAGMENT_ROWS;
  constexpr ushort VALUE_FRAGMENTS = BV / FragmentOps::FRAGMENT_COLS;
  using InputType = metal::conditional_t<USE_MXU, T, float>;
  using TileFragment = Fragment<float, TOKEN_FRAGMENTS, VALUE_FRAGMENTS, FragmentOps>;
  using KeyFragment = OperandFragment<InputType, TOKEN_FRAGMENTS, 1, FragmentOps>;
  using H0Fragment = OperandFragment<InputType, 1, VALUE_FRAGMENTS, FragmentOps, ReadTranspose>;
  using BlockMatrixFragment = OperandFragment<float, TOKEN_FRAGMENTS, TOKEN_FRAGMENTS, FragmentOps>;
  using ValueTileFragment = OperandFragment<float, TOKEN_FRAGMENTS, VALUE_FRAGMENTS, FragmentOps>;

  static_assert(BT == 16, "GdnTreeUpdateSolve expects BT=16");
  static_assert(BT % FragmentOps::FRAGMENT_ROWS == 0, "BT must be a multiple of the fragment row size");
  static_assert(BV % FragmentOps::FRAGMENT_COLS == 0, "BV must be a multiple of the fragment column size");
  static_assert(HEAD_K_DIM % FragmentOps::FRAGMENT_ROWS == 0, "HEAD_K_DIM must fit fragment rows");

  const uint batch_value_head_idx = batch_idx * num_v_heads + value_head_idx;

  const uint value_heads_per_key_head = num_v_heads / num_k_heads;
  const uint key_head_idx = value_head_idx / value_heads_per_key_head;

  const uint value_dim_base = value_tile_idx * BV;
  const uint tile_value_cols = min(BV, head_v_dim - value_dim_base);
  const uint num_blocks = (tree_size + BT - 1u) / BT;
  const int h0_slot = h0_idx[batch_idx];

  const uint key_token_stride = num_k_heads * HEAD_K_DIM;
  const uint value_token_stride = num_v_heads * head_v_dim;
  const uint scalar_token_stride = num_v_heads;

  const device T* key_head = k + (batch_idx * tree_size * num_k_heads + key_head_idx) * HEAD_K_DIM;
  const device T* value_head_tile =
      v + (batch_idx * tree_size * num_v_heads + value_head_idx) * head_v_dim + value_dim_base;
  const device float* prefix_head = prefix + batch_idx * tree_size * num_v_heads + value_head_idx;
  const device float* beta_head = beta + batch_idx * tree_size * num_v_heads + value_head_idx;
  const device float* a_matrix = a + batch_value_head_idx * tree_size * tree_size;
  const device float* a_inv_matrix = a_inv + batch_value_head_idx * tree_size * tree_size;
  device float* u_head_tile = u + batch_value_head_idx * tree_size * head_v_dim + value_dim_base;
  const device T* h0_head_tile =
      h0_slot >= 0
          ? h0 + (uint(h0_slot) * num_v_heads + value_head_idx) * head_v_dim * HEAD_K_DIM + value_dim_base * HEAD_K_DIM
          : nullptr;

  for (uint block_idx = 0; block_idx < num_blocks; ++block_idx) {
    const uint row_base = block_idx * BT;
    const uint tile_rows = min(BT, tree_size - row_base);

    TileFragment acc;
    acc.clear();

    if (h0_slot >= 0) {
      for (uint key_dim_base = 0; key_dim_base < HEAD_K_DIM; key_dim_base += FragmentOps::FRAGMENT_ROWS) {
        KeyFragment key_frag;
        H0Fragment h0_frag;

        const device T* key_block = key_head + row_base * key_token_stride + key_dim_base;
        const device T* h0_block = h0_head_tile + key_dim_base;

        if (tile_rows == BT) {
          key_frag.load_from(lane_idx, fragment_source(key_block, key_token_stride));
        } else {
          key_frag.load_from(
              lane_idx,
              fragment_source(key_block, key_token_stride).bounded(tile_rows, FragmentOps::FRAGMENT_ROWS)
          );
        }

        if (tile_value_cols == BV) {
          h0_frag.load_from(lane_idx, fragment_source(h0_block, HEAD_K_DIM));
        } else {
          h0_frag.load_from(
              lane_idx,
              fragment_source(h0_block, HEAD_K_DIM).bounded(tile_value_cols, FragmentOps::FRAGMENT_ROWS)
          );
        }

        fragment_mma(acc, key_frag, h0_frag);
      }
    }

    acc.map_coords(lane_idx, [&](ushort local_row_idx, ushort local_value_col_idx, float kh0_value) {
      const uint token_idx = row_base + local_row_idx;
      const bool valid_output = local_row_idx < tile_rows && local_value_col_idx < tile_value_cols;
      if (!valid_output) {
        return 0.0f;
      }

      const float value = float(value_head_tile[token_idx * value_token_stride + local_value_col_idx]);
      const float prefix_value = prefix_head[token_idx * scalar_token_stride];
      const float beta_value = beta_head[token_idx * scalar_token_stride];
      return beta_value * (value - exp(prefix_value) * kh0_value);
    });

    for (uint prev_block_idx = 0; prev_block_idx < block_idx; ++prev_block_idx) {
      const uint prev_row_base = prev_block_idx * BT;
      const uint prev_tile_rows = min(BT, tree_size - prev_row_base);
      BlockMatrixFragment a_frag;
      ValueTileFragment u_prev_frag;

      const device float* a_block = a_matrix + row_base * tree_size + prev_row_base;
      const device float* u_prev_block = u_head_tile + prev_row_base * head_v_dim;
      a_frag.load_from(lane_idx, fragment_source(a_block, tree_size).bounded(tile_rows, prev_tile_rows));
      u_prev_frag.load_from(
          lane_idx,
          fragment_source(u_prev_block, head_v_dim).bounded(prev_tile_rows, tile_value_cols)
      );
      a_frag.map([](float value) { return -value; });
      fragment_mma(acc, a_frag, u_prev_frag);
    }

    BlockMatrixFragment inv_frag;
    TileFragment solved;
    solved.clear();

    const device float* inv_block = a_inv_matrix + row_base * tree_size + row_base;
    inv_frag.load_from(lane_idx, fragment_source(inv_block, tree_size).bounded(tile_rows, tile_rows));
    fragment_mma(solved, inv_frag, acc);
    solved.store_safe(lane_idx, u_head_tile + row_base * head_v_dim, head_v_dim, short2(tile_value_cols, tile_rows));

    threadgroup_barrier(mem_flags::mem_device);
  }
}
