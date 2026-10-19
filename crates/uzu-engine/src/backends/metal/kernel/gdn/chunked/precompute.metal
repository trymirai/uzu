#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../common/thread_context.h"
#include "../common/gram.h"
#include "../common/tri_inv.h"

using namespace metal;
using namespace uzu::matmul;

#define BLOCK_SIZE 16u
#define COL_TILE 32u
#define MAX_BLOCKS 4u
#define A_TILE_COUNT 6u
#define MAX_GROUPS 4u
#define PRECOMPUTE_THREADS 128

// q_norm:        [suffix_len, key_dim].
// k_norm:        [suffix_len, key_dim].
// log_decay:     [suffix_len, num_v_heads].
// beta:          [suffix_len, num_v_heads].
// g_out:         [num_v_heads, suffix_len] chunk-local cumulative log decay.
// qk_scaled_out: [chunks, num_v_heads, CHUNK, CHUNK] decay-scaled causal Q K^T.
// t_out:         [chunks, num_v_heads, CHUNK, CHUNK] dense T = (I + A)^-1, bf16.
//
// One threadgroup owns one (chunk, k-head); simdgroup b owns block row b. Per grouped v-head:
// 1. g = prefix sum of log_decay.
// 2. qk_scaled = exp(g_r - g_c) * Q K^T for c <= r.
// 3. A = beta_r * exp(g_r - g_c) * K K^T for c < r.
// 4. T = (I + A)^-1 by block forward substitution.
template <uint HEAD_K_DIM, uint CHUNK_SIZE>
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 64)
KERNEL(DeltaNetChunkedPrecompute)(
    device const float* q_norm,
    device const float* k_norm,
    device const float* log_decay,
    device const float* beta,
    device float* g_out,
    device float* qk_scaled_out,
    device bfloat* t_out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& key_dim,
    constant const uint& suffix_len,
    threadgroup float a_tiles[A_TILE_COUNT * BLOCK_SIZE * COL_TILE],
    threadgroup float diag_inverses[MAX_BLOCKS * BLOCK_SIZE * BLOCK_SIZE],
    threadgroup float g_tiles[MAX_GROUPS * CHUNK_SIZE],
    threadgroup float beta_tiles[MAX_GROUPS * CHUNK_SIZE],
    const ThreadContext thread_context,
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hk_idx GROUPS(num_k_heads),
    const uint thread_idx THREADS(PRECOMPUTE_THREADS)
) {
  using Ops = SimdgroupFragmentOps;
  constexpr ushort ROW_FRAGMENTS = BLOCK_SIZE / Ops::FRAGMENT_ROWS;
  constexpr ushort GRAM_COL_FRAGMENTS = COL_TILE / Ops::FRAGMENT_COLS;
  using GramAccFragment = Fragment<float, ROW_FRAGMENTS, GRAM_COL_FRAGMENTS, Ops>;
  using GramLeftFragment = OperandFragment<float, ROW_FRAGMENTS, 1, Ops>;
  using GramRightFragment = OperandFragment<float, 1, GRAM_COL_FRAGMENTS, Ops, ReadTranspose>;

  using TileFragment = Fragment<float, ROW_FRAGMENTS, ROW_FRAGMENTS, Ops>;
  using MatrixFragment = OperandFragment<float, ROW_FRAGMENTS, ROW_FRAGMENTS, Ops>;
  using ValueFragment = OperandFragment<bfloat, ROW_FRAGMENTS, ROW_FRAGMENTS, Ops>;

  static_assert(HEAD_K_DIM % Ops::FRAGMENT_ROWS == 0, "HEAD_K_DIM must align to fragment K");
  constexpr uint num_blocks = CHUNK_SIZE / BLOCK_SIZE;
  static_assert(num_blocks == MAX_BLOCKS, "one simdgroup per block row");

  const uint lane = thread_context.simd_lane_id;
  const uint block_idx = thread_context.simdgroup_index;
  // First A tile of block row b in threadgroup memory: rows hold b / 2 + 1 tiles.
  auto tile_base = [](uint b) UZU_ALWAYS_INLINE { return b < 2 ? b : 2 * b - 2; };

  const uint chunk_token_base = chunk_idx * CHUNK_SIZE;
  const uint valid_tokens = min(uint(CHUNK_SIZE), suffix_len - chunk_token_base);
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint row_base = block_idx * BLOCK_SIZE;
  const uint row_token_base = chunk_token_base + row_base;
  const uint valid_rows = valid_tokens > row_base ? min(BLOCK_SIZE, valid_tokens - row_base) : 0u;
  const uint num_col_tiles = block_idx / 2 + 1;

  // Simdgroup b loads log_decay and beta of grouped v-head b ahead of the MMA loop.
  constexpr uint decay_passes = CHUNK_SIZE / METAL_SIMD_SIZE;
  const bool has_group = block_idx < groups_per_head;
  const uint group_hv_idx = hk_idx * groups_per_head + block_idx;
  float decay_values[decay_passes];
  float beta_values[decay_passes];
  METAL_PRAGMA_UNROLL
  for (uint pass = 0; pass < decay_passes; ++pass) {
    const uint local = pass * METAL_SIMD_SIZE + lane;
    const bool valid = has_group && local < valid_tokens;
    const uint offset = (chunk_token_base + local) * num_v_heads + group_hv_idx;
    decay_values[pass] = valid ? log_decay[offset] : 0.0f;
    beta_values[pass] = valid ? beta[offset] : 0.0f;
  }

  // K K^T and Q K^T of this block row up to the diagonal; tail tiles stay zero.
  GramAccFragment kk_acc[2];
  GramAccFragment qk_acc[2];
  METAL_PRAGMA_UNROLL
  for (uint c = 0; c < 2; ++c) {
    kk_acc[c].clear();
    qk_acc[c].clear();
  }
  if (valid_rows != 0) {
    for (uint k_block_start = 0; k_block_start < HEAD_K_DIM; k_block_start += Ops::FRAGMENT_ROWS) {
      const device float* k_rows = k_norm + row_token_base * key_dim + hk_idx * HEAD_K_DIM + k_block_start;
      const device float* q_rows = q_norm + row_token_base * key_dim + hk_idx * HEAD_K_DIM + k_block_start;
      METAL_PRAGMA_UNROLL
      for (uint c = 0; c < 2; ++c) {
        const uint col_base = c * COL_TILE;
        const uint col_token_base = chunk_token_base + col_base;
        const uint tile_cols = min(uint(COL_TILE), uint(CHUNK_SIZE - col_base));
        const uint valid_cols = valid_tokens > col_base ? min(tile_cols, valid_tokens - col_base) : 0u;
        if (c < num_col_tiles && valid_cols != 0) {
          const device float* k_cols = k_norm + col_token_base * key_dim + hk_idx * HEAD_K_DIM + k_block_start;
          accumulate_dual_gram_tile<GramAccFragment, GramLeftFragment, GramRightFragment>(
              kk_acc[c],
              qk_acc[c],
              k_rows,
              q_rows,
              k_cols,
              int(key_dim),
              ushort(valid_rows),
              ushort(valid_cols),
              Ops::FRAGMENT_ROWS,
              valid_rows == BLOCK_SIZE,
              valid_cols == tile_cols,
              ushort(lane)
          );
        }
      }
    }
  }

  if (has_group) {
    float carry = 0.0f;
    METAL_PRAGMA_UNROLL
    for (uint pass = 0; pass < decay_passes; ++pass) {
      const uint local = pass * METAL_SIMD_SIZE + lane;
      const float prefix = carry + simd_prefix_inclusive_sum(decay_values[pass]);
      g_tiles[block_idx * CHUNK_SIZE + local] = prefix;
      beta_tiles[block_idx * CHUNK_SIZE + local] = beta_values[pass];
      if (local < valid_tokens) {
        g_out[group_hv_idx * suffix_len + chunk_token_base + local] = prefix;
      }
      carry += simd_sum(decay_values[pass]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float* own_tiles = a_tiles + tile_base(block_idx) * (BLOCK_SIZE * COL_TILE);

  for (uint group = 0; group < groups_per_head; ++group) {
    const threadgroup float* g_tile = g_tiles + group * CHUNK_SIZE;
    const uint head_base = chunk_idx * num_v_heads + hk_idx * groups_per_head + group;
    METAL_PRAGMA_UNROLL
    for (uint c = 0; c < 2; ++c) {
      if (c < num_col_tiles) {
        const uint col_base = c * COL_TILE;
        const short2 tile_dims = short2(min(uint(COL_TILE), uint(CHUNK_SIZE - col_base)), BLOCK_SIZE);
        GramAccFragment scaled = qk_acc[c];
        scaled.map_coords(lane, [&](short r, short col_in_tile, float value) UZU_ALWAYS_INLINE {
          const uint row = row_base + uint(r);
          const uint col = col_base + uint(col_in_tile);
          if (row >= valid_tokens || col >= valid_tokens || col > row) {
            return 0.0f;
          }
          return value * fast::exp(g_tile[row] - g_tile[col]);
        });
        scaled.store_safe(
            lane,
            qk_scaled_out + head_base * CHUNK_SIZE * CHUNK_SIZE + row_base * CHUNK_SIZE + col_base,
            CHUNK_SIZE,
            tile_dims
        );
      }
    }
  }

  for (uint group = 0; group < groups_per_head; ++group) {
    const threadgroup float* g_tile = g_tiles + group * CHUNK_SIZE;
    const threadgroup float* beta_tile = beta_tiles + group * CHUNK_SIZE;
    const uint head_base = chunk_idx * num_v_heads + hk_idx * groups_per_head + group;

    METAL_PRAGMA_UNROLL
    for (uint c = 0; c < 2; ++c) {
      if (c < num_col_tiles) {
        const uint col_base = c * COL_TILE;
        GramAccFragment a = kk_acc[c];
        a.map_coords(lane, [&](short r, short col_in_tile, float value) UZU_ALWAYS_INLINE {
          const uint row = row_base + uint(r);
          const uint col = col_base + uint(col_in_tile);
          if (row >= valid_tokens || col >= valid_tokens || col >= row) {
            return 0.0f;
          }
          return beta_tile[row] * fast::exp(g_tile[row] - g_tile[col]) * value;
        });
        a.store(lane, own_tiles + c * (BLOCK_SIZE * COL_TILE), COL_TILE);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    invert_lower_triangular_block<BLOCK_SIZE, COL_TILE>(
        diag_inverses + block_idx * (BLOCK_SIZE * BLOCK_SIZE),
        own_tiles + (block_idx / 2) * (BLOCK_SIZE * COL_TILE) + (block_idx % 2) * BLOCK_SIZE,
        valid_rows,
        lane
    );

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Solve this simdgroup's 16-column slice of T; blocks above the slice are zero, solved blocks stay in registers.
    const uint slice_base = block_idx * BLOCK_SIZE;
    device bfloat* t_head = t_out + head_base * CHUNK_SIZE * CHUNK_SIZE + slice_base;
    ValueFragment t_blocks[num_blocks];
    METAL_PRAGMA_UNROLL
    for (uint b = 0; b < num_blocks; ++b) {
      TileFragment t_block;
      t_block.clear();
      if (b >= block_idx) {
        const threadgroup float* b_tiles = a_tiles + tile_base(b) * (BLOCK_SIZE * COL_TILE);
        TileFragment acc_t;
        acc_t.map_coords(lane, [&](short local_row, short local_col, float) UZU_ALWAYS_INLINE {
          return b * BLOCK_SIZE + uint(local_row) == slice_base + uint(local_col) ? 1.0f : 0.0f;
        });
        METAL_PRAGMA_UNROLL
        for (uint prev_block_idx = 0; prev_block_idx < b; ++prev_block_idx) {
          if (prev_block_idx >= block_idx) {
            MatrixFragment a_frag;
            a_frag.load_from(
                lane,
                fragment_source(
                    b_tiles + (prev_block_idx / 2) * (BLOCK_SIZE * COL_TILE) + (prev_block_idx % 2) * BLOCK_SIZE,
                    COL_TILE
                )
            );
            a_frag.map([](float value) UZU_ALWAYS_INLINE { return -value; });
            fragment_mma(acc_t, a_frag, t_blocks[prev_block_idx]);
          }
        }
        MatrixFragment inv_frag;
        inv_frag.load_from(
            lane,
            fragment_source(
                static_cast<const threadgroup float*>(diag_inverses + b * (BLOCK_SIZE * BLOCK_SIZE)),
                BLOCK_SIZE
            )
        );
        fragment_mma(t_block, inv_frag, acc_t);
      }
      t_block.store(lane, t_head + b * BLOCK_SIZE * CHUNK_SIZE, CHUNK_SIZE);
      METAL_PRAGMA_UNROLL
      for (ushort e = 0; e < ValueFragment::ELEMENTS_PER_FRAGMENT; ++e) {
        t_blocks[b].elements()[e] = bfloat(t_block.elements()[e]);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
