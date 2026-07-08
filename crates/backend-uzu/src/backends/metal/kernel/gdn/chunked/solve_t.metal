#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/mxu_fragment_ops.h"
#include "../../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;

#define SOLVE_T_BLOCK 16u

// Builds dense T = (I + A)^-1 as bf16 for mega apply. This is block forward
// substitution over a_packed plus the per-block inverses from Solve.
template <uint CHUNK_SIZE, uint BV>
VARIANTS(CHUNK_SIZE, 16, 32, 64)
VARIANTS(BV, 16, 32)
KERNEL(DeltaNetChunkedSolveT)(
    device const float* a_packed,
    device const float* a_inv,
    device bfloat* t_out,
    constant const uint& num_v_heads,
    constant const uint& suffix_len,
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint tile_idx GROUPS(CHUNK_SIZE.div_ceil(BV)),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  using Ops = SimdgroupFragmentOps;
  using InputType = float;
  constexpr ushort TOKEN_FRAGMENTS = SOLVE_T_BLOCK / Ops::FRAGMENT_ROWS;
  constexpr ushort VALUE_FRAGMENTS = BV / Ops::FRAGMENT_COLS;
  static_assert(BV % Ops::FRAGMENT_COLS == 0, "BV must align to fragment columns");
  using TileFragment = Fragment<float, TOKEN_FRAGMENTS, VALUE_FRAGMENTS, Ops>;
  using MatrixFragment = OperandFragment<InputType, TOKEN_FRAGMENTS, TOKEN_FRAGMENTS, Ops>;
  using ValueFragment = OperandFragment<InputType, TOKEN_FRAGMENTS, VALUE_FRAGMENTS, Ops>;
  using PairMatrixFragment = OperandFragment<InputType, TOKEN_FRAGMENTS, 2 * TOKEN_FRAGMENTS, Ops>;
  using PairValueFragment = OperandFragment<InputType, 2 * TOKEN_FRAGMENTS, VALUE_FRAGMENTS, Ops>;

  constexpr uint num_blocks = (CHUNK_SIZE + SOLVE_T_BLOCK - 1) / SOLVE_T_BLOCK;
  constexpr uint num_col_pairs = (num_blocks + 1) / 2;
  const uint slice_base = tile_idx * BV;
  const uint tile_cols = min(BV, CHUNK_SIZE - slice_base);
  const uint token_base = chunk_idx * CHUNK_SIZE;

  const device float* a_blocks =
      a_packed + (chunk_idx * num_v_heads + hv_idx) * num_blocks * num_col_pairs * (SOLVE_T_BLOCK * 2 * SOLVE_T_BLOCK);
  const device float* inv_blocks =
      a_inv + (chunk_idx * num_v_heads + hv_idx) * num_blocks * (SOLVE_T_BLOCK * SOLVE_T_BLOCK);
  device bfloat* t_head = t_out + (chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE * CHUNK_SIZE + slice_base;

  for (uint block_idx = 0; block_idx < num_blocks; ++block_idx) {
    const uint row_base = block_idx * SOLVE_T_BLOCK;

    // -- RHS = identity slice: acc[row, col] = (row_base+row == slice_base+col) -
    TileFragment acc_t;
    acc_t.clear();
    acc_t.map_coords(lane, [&](short local_row, short local_col, float) {
      const uint global_row = row_base + uint(local_row);
      const uint global_col = slice_base + uint(local_col);
      return global_row == global_col ? 1.0f : 0.0f;
    });

    // -- Forward substitution: subtract A . T_prev (T_prev in bf16, own slice) --
    const uint num_full_pairs = block_idx / 2;
    for (uint pair_idx = 0; pair_idx < num_full_pairs; ++pair_idx) {
      PairMatrixFragment a_frag;
      const device float* a_pair =
          a_blocks + (block_idx * num_col_pairs + pair_idx) * (SOLVE_T_BLOCK * 2 * SOLVE_T_BLOCK);
      a_frag.load_from(lane, fragment_source(a_pair, 2 * SOLVE_T_BLOCK));
      a_frag.map([](float value) { return -value; });

      PairValueFragment t_prev_frag;
      const device bfloat* t_prev = t_head + pair_idx * 2 * SOLVE_T_BLOCK * CHUNK_SIZE;
      t_prev_frag.load_from(lane, fragment_source(t_prev, CHUNK_SIZE).bounded(2 * SOLVE_T_BLOCK, tile_cols));
      fragment_mma(acc_t, a_frag, t_prev_frag);
    }

    for (uint prev_block_idx = num_full_pairs * 2; prev_block_idx < block_idx; ++prev_block_idx) {
      MatrixFragment a_frag;
      const device float* a_block =
          a_blocks + (block_idx * num_col_pairs + prev_block_idx / 2) * (SOLVE_T_BLOCK * 2 * SOLVE_T_BLOCK) +
          (prev_block_idx % 2) * SOLVE_T_BLOCK;
      a_frag.load_from(lane, fragment_source(a_block, 2 * SOLVE_T_BLOCK));
      a_frag.map([](float value) { return -value; });

      ValueFragment t_prev_frag;
      const device bfloat* t_prev = t_head + prev_block_idx * SOLVE_T_BLOCK * CHUNK_SIZE;
      t_prev_frag.load_from(lane, fragment_source(t_prev, CHUNK_SIZE).bounded(SOLVE_T_BLOCK, tile_cols));
      fragment_mma(acc_t, a_frag, t_prev_frag);
    }

    // -- Apply the diagonal-block inverse and store T ------------------------
    MatrixFragment inv_frag;
    inv_frag.load_from(lane, fragment_source(inv_blocks + block_idx * SOLVE_T_BLOCK * SOLVE_T_BLOCK, SOLVE_T_BLOCK));

    TileFragment solved_t;
    solved_t.clear();
    fragment_mma(solved_t, inv_frag, acc_t);
    solved_t.store_safe(lane, t_head + row_base * CHUNK_SIZE, CHUNK_SIZE, short2(tile_cols, SOLVE_T_BLOCK));

    simdgroup_barrier(mem_flags::mem_device);
  }

  // Suppress unused warnings when suffix_len bounds are not needed on the fast
  // path (invalid rows carry identity via a_inv, matching BuildWU).
  (void)token_base;
  (void)suffix_len;
}
