#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/mxu_fragment_ops.h"
#include "../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;

// Block forward substitution for (I + A) U = beta * (v - exp(prefix) * kh0).
// Per 16-token block i, serially: acc_i = beta * (v_i - exp(prefix_i) * kh0_i);
// acc_i -= A[i, j] @ U[j] for all j < i; U[i] = Ainv[i] @ acc_i.
//
// Shapes (NB = ceil(T/16)):
//   kh0    [B, T, HV, head_v_dim] f32, from BuildTreeGram (absent when !use_h0)
//   v      [B, T, HV, head_v_dim] in T
//   prefix, beta [B, T, HV] f32
//   A      [B*HV, NB, ceil(NB/2), 16, 32] f32 packed pair tiles from BuildTreeGram
//          (block (i,j) at pair j/2, column (j%2)*16)
//   Ainv   [B*HV, NB, 16, 16] f32 diagonal-block inverses from BuildTreeGram
//   h0_idx [B] i32, negative = zero initial state (absent when !use_h0)
//   U      [B*HV, T, head_v_dim] f32 (output)
//
// Grid: one simdgroup per (batch, value head, BV value-dim tile); the history is
// consumed one packed pair tile ([16,32] A x [32,BV] U) per fragment_mma plus a
// single-block tail when i is odd.
// A and Ainv are kept in f32: the (I + A)^-1 forward-substitution cascade is
// precision-sensitive.
template <typename T, uint BV, bool USE_MXU>
VARIANTS(T, float, bfloat)
VARIANTS(BV, 16, 32)
VARIANTS(USE_MXU, false, true)
CONSTRAINT(!USE_MXU || T != "float")
CONSTRAINT(!USE_MXU || BV >= 32)
PUBLIC KERNEL(TreeUpdateSolve)(
    const device float* kh0 OPTIONAL(use_h0),
    const device T* v,
    const device float* prefix,
    const device float* beta,
    const device float* a_packed,
    const device float* a_inv,
    const device int* h0_idx OPTIONAL(use_h0),
    device float* u,
    constant const uint& batch_size,
    constant const uint& tree_size,
    constant const uint& value_heads,
    constant const uint& head_v_dim,
    const bool use_h0 SPECIALIZE,
    const uint batch_idx GROUPS(batch_size),
    const uint value_head_idx GROUPS(value_heads),
    const uint value_tile_idx GROUPS(head_v_dim.div_ceil(BV)),
    const uint lane_idx THREADS(METAL_SIMD_SIZE)
) {
  constexpr uint BT = 16;
  using FragmentOps = metal::conditional_t<USE_MXU, MxuFragmentOps<>, SimdgroupFragmentOps>;
  constexpr ushort TOKEN_FRAGMENTS = BT / FragmentOps::FRAGMENT_ROWS;
  constexpr ushort VALUE_FRAGMENTS = BV / FragmentOps::FRAGMENT_COLS;
  using TileFragment = Fragment<float, TOKEN_FRAGMENTS, VALUE_FRAGMENTS, FragmentOps>;
  using BlockMatrixFragment = OperandFragment<float, TOKEN_FRAGMENTS, TOKEN_FRAGMENTS, FragmentOps>;
  using ValueTileFragment = OperandFragment<float, TOKEN_FRAGMENTS, VALUE_FRAGMENTS, FragmentOps>;
  using PairMatrixFragment = OperandFragment<float, TOKEN_FRAGMENTS, 2 * TOKEN_FRAGMENTS, FragmentOps>;
  using PairValueFragment = OperandFragment<float, 2 * TOKEN_FRAGMENTS, VALUE_FRAGMENTS, FragmentOps>;

  static_assert(BV % FragmentOps::FRAGMENT_COLS == 0, "BV must be a multiple of the fragment column size");

  const uint batch_value_head_idx = batch_idx * value_heads + value_head_idx;

  const uint value_dim_base = value_tile_idx * BV;
  const uint tile_value_cols = min(BV, head_v_dim - value_dim_base);
  const uint num_blocks = div_ceil(tree_size, BT);
  const int h0_slot = use_h0 ? h0_idx[batch_idx] : -1;

  const uint value_token_stride = value_heads * head_v_dim;
  const uint scalar_token_stride = value_heads;

  const device float* kh0_head_tile =
      kh0 + (batch_idx * tree_size * value_heads + value_head_idx) * head_v_dim + value_dim_base;
  const device T* value_head_tile =
      v + (batch_idx * tree_size * value_heads + value_head_idx) * head_v_dim + value_dim_base;
  const device float* prefix_head = prefix + batch_idx * tree_size * value_heads + value_head_idx;
  const device float* beta_head = beta + batch_idx * tree_size * value_heads + value_head_idx;
  const uint num_col_pairs = div_ceil(num_blocks, 2u);
  const device float* a_blocks = a_packed + batch_value_head_idx * num_blocks * num_col_pairs * (BT * 2 * BT);
  const device float* a_inv_blocks = a_inv + batch_value_head_idx * num_blocks * (BT * BT);
  device float* u_head_tile = u + batch_value_head_idx * tree_size * head_v_dim + value_dim_base;

  for (uint block_idx = 0; block_idx < num_blocks; ++block_idx) {
    const uint row_base = block_idx * BT;
    const uint tile_rows = min(BT, tree_size - row_base);

    TileFragment acc;

    if (h0_slot >= 0) {
      const device float* kh0_block = kh0_head_tile + row_base * value_token_stride;
      acc.load_maybe_bounded(
          lane_idx,
          kh0_block,
          value_token_stride,
          tile_rows == BT && tile_value_cols == BV,
          tile_rows,
          tile_value_cols
      );
    } else {
      acc.clear();
    }

    // prefix/beta/exp are per-row: one lane computes them, consumers read via simd_shuffle.
    float lane_beta = 0.0f;
    float lane_decay = 0.0f;
    if (lane_idx < tile_rows) {
      const uint token_idx = row_base + lane_idx;
      lane_beta = beta_head[token_idx * scalar_token_stride];
      lane_decay = exp(prefix_head[token_idx * scalar_token_stride]);
    }

    TileFragment v_frag;
    const device T* v_block = value_head_tile + row_base * value_token_stride;
    v_frag.load_maybe_bounded(
        lane_idx,
        v_block,
        value_token_stride,
        tile_rows == BT && tile_value_cols == BV,
        tile_rows,
        tile_value_cols
    );

    TileFragment::zip_for_each_coord(
        lane_idx,
        [&](ushort local_row_idx, ushort, thread float& acc_value, thread float& v_value) {
          const float beta_row = simd_shuffle(lane_beta, local_row_idx);
          const float decay_row = simd_shuffle(lane_decay, local_row_idx);
          acc_value = beta_row * (v_value - decay_row * acc_value);
        },
        acc,
        v_frag
    );

    const uint num_full_pairs = block_idx / 2;
    for (uint pair_idx = 0; pair_idx < num_full_pairs; ++pair_idx) {
      PairMatrixFragment a_frag;
      PairValueFragment u_prev_frag;

      const device float* a_pair = a_blocks + (block_idx * num_col_pairs + pair_idx) * (BT * 2 * BT);
      const device float* u_prev_block = u_head_tile + pair_idx * 2 * BT * head_v_dim;
      const bool full = tile_rows == BT && tile_value_cols == BV;
      a_frag.load_maybe_bounded(lane_idx, a_pair, 2 * BT, full, tile_rows, 2 * BT);
      u_prev_frag.load_maybe_bounded(lane_idx, u_prev_block, head_v_dim, full, 2 * BT, tile_value_cols);
      a_frag.map([](float value) { return -value; });
      fragment_mma(acc, a_frag, u_prev_frag);
    }

    for (uint prev_block_idx = num_full_pairs * 2; prev_block_idx < block_idx; ++prev_block_idx) {
      const uint prev_row_base = prev_block_idx * BT;
      BlockMatrixFragment a_frag;
      ValueTileFragment u_prev_frag;

      const device float* a_block =
          a_blocks + (block_idx * num_col_pairs + prev_block_idx / 2) * (BT * 2 * BT) + (prev_block_idx % 2) * BT;
      a_frag.load_from(lane_idx, fragment_source(a_block, 2 * BT).bounded(tile_rows, BT));
      const device float* u_prev_block = u_head_tile + prev_row_base * head_v_dim;
      u_prev_frag.load_from(lane_idx, fragment_source(u_prev_block, head_v_dim).bounded(BT, tile_value_cols));
      a_frag.map([](float value) { return -value; });
      fragment_mma(acc, a_frag, u_prev_frag);
    }

    BlockMatrixFragment inv_frag;
    TileFragment solved;
    solved.clear();

    const device float* inv_block = a_inv_blocks + block_idx * (BT * BT);
    inv_frag.load_from(lane_idx, fragment_source(inv_block, BT).bounded(tile_rows, tile_rows));
    fragment_mma(solved, inv_frag, acc);
    solved.store_safe(lane_idx, u_head_tile + row_base * head_v_dim, head_v_dim, short2(tile_value_cols, tile_rows));

    // The threadgroup is a single simdgroup, so the block-serial U dependency
    // only needs a simdgroup-scoped device-memory fence.
    simdgroup_barrier(mem_flags::mem_device);
  }
}
