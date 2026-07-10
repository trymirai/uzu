#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../common/thread_context.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/mxu_fragment_ops.h"
#include "../../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;

#define OUTPUT_STATE_THREADS 128
#define OUTPUT_STATE_NUM_SIMDGROUPS (OUTPUT_STATE_THREADS / METAL_SIMD_SIZE)
#define OUTPUT_STATE_CHUNK 64
#define OUTPUT_STATE_KEY_TILE (HEAD_K_DIM / OUTPUT_STATE_NUM_SIMDGROUPS)

// q_norm:    [suffix_len, key_dim].
// k_norm:    [suffix_len, key_dim].
// in_proj:   [suffix_len, total_proj_dim]; V starts at offset 2*key_dim.
// qk_scaled: [chunks, num_v_heads, CHUNK, CHUNK] from Gram.
// t_mat:     [chunks, num_v_heads, CHUNK, CHUNK] dense causal inverse, bf16.
// g:         [num_v_heads, suffix_len] cumulative log decay.
// beta:      [suffix_len, num_v_heads].
// state:     [num_v_heads, head_v_dim, HEAD_K_DIM], f32 input/output.
// out:       [suffix_len, value_dim].
// ScratchT:  R/Vnew threadgroup storage precision.
//
// One threadgroup owns one (v-head, value-slice). It loads the state slice as
// S^T in threadgroup memory, then scans chunks:
// 1. R = beta * (V - exp(g) * K * S^T).
// 2. Vnew = T * R.
// 3. out = exp(g) * Q * S^T + qk_scaled * Vnew.
// 4. S^T = exp(g_last) * S^T + (exp(g_last - g) * K)^T * Vnew.
template <typename T, typename O, typename ScratchT, uint HEAD_K_DIM, uint VT, bool USE_MXU>
VARIANTS(T, float, bfloat)
VARIANTS(O, float, bfloat)
VARIANTS(ScratchT, float, bfloat)
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(VT, 32)
VARIANTS(USE_MXU, false, true)
KERNEL(DeltaNetChunkedOutputAndState)(
    device const float* q_norm,
    device const float* k_norm,
    device const T* in_proj,
    device const float* qk_scaled,
    device const bfloat* t_mat,
    device const float* g,
    device const float* beta,
    device float* state,
    device O* out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const uint& suffix_len,
    threadgroup float transposed_state_tile[HEAD_K_DIM * VT],
    threadgroup ScratchT residual_new_value_tile[OUTPUT_STATE_CHUNK * VT],
    const ThreadContext thread_context,
    const uint hv_idx GROUPS(num_v_heads),
    const uint value_tile_idx GROUPS(head_v_dim.div_ceil(VT)),
    const uint thread_idx THREADS(OUTPUT_STATE_THREADS)
) {
  using Ops = metal::conditional_t<USE_MXU, MxuFragmentOps<>, SimdgroupFragmentOps>;
  constexpr ushort FR = Ops::FRAGMENT_ROWS;
  constexpr ushort FC = Ops::FRAGMENT_COLS;
  static_assert(HEAD_K_DIM % FR == 0, "HEAD_K_DIM must be a multiple of fragment rows");
  static_assert(OUTPUT_STATE_CHUNK % FR == 0, "chunk size must be a multiple of fragment rows");
  static_assert(VT % FC == 0, "VT must be a multiple of fragment columns");
  static_assert(OUTPUT_STATE_KEY_TILE % FC == 0, "key tile must be a multiple of fragment columns");

  constexpr uint TOKEN_TILE = (VT >= 32) ? 16u : 32u;
  constexpr uint NUM_TOKEN_TILES = OUTPUT_STATE_CHUNK / TOKEN_TILE;
  constexpr ushort TOKEN_FRAGMENTS = TOKEN_TILE / FR;
  constexpr ushort VALUE_FRAGMENTS = VT / FC;
  constexpr ushort KEY_FRAGMENTS = OUTPUT_STATE_KEY_TILE / FC;

  using TokenValueFragment = Fragment<float, TOKEN_FRAGMENTS, VALUE_FRAGMENTS, Ops>;
  using TokenKeyFragment = OperandFragment<float, TOKEN_FRAGMENTS, 1, Ops>;
  using KeyValueFragment = OperandFragment<float, 1, VALUE_FRAGMENTS, Ops>;
  using ScratchKeyValueFragment = OperandFragment<ScratchT, 1, VALUE_FRAGMENTS, Ops>;
  using ValueKeyFragment = Fragment<float, VALUE_FRAGMENTS, KEY_FRAGMENTS, Ops>;
  using ScratchValueTokenFragment = OperandFragment<ScratchT, VALUE_FRAGMENTS, 1, Ops>;
  using TokenKeyTileFragment = OperandFragment<float, 1, KEY_FRAGMENTS, Ops>;

  const uint lane = thread_context.simd_lane_id;
  const uint simdgroup_idx = thread_context.simdgroup_index;
  const uint value_base = value_tile_idx * VT;
  if (value_base >= head_v_dim) {
    return;
  }
  const uint hk_idx = hv_idx / (num_v_heads / num_k_heads);
  const uint num_chunks = (suffix_len + OUTPUT_STATE_CHUNK - 1) / OUTPUT_STATE_CHUNK;
  const uint conv_dim = 2 * key_dim + value_dim;
  const uint total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
  const device float* g_head = g + hv_idx * suffix_len;

  // Load the initial state slice transposed into threadgroup memory:
  //   transposed_state_tile[key * VT + value] = state[(hv, value_base + value, key)]
  for (uint idx = thread_idx; idx < VT * HEAD_K_DIM; idx += OUTPUT_STATE_THREADS) {
    const uint local_value_idx = idx / HEAD_K_DIM;
    const uint local_key_idx = idx - local_value_idx * HEAD_K_DIM;
    transposed_state_tile[local_key_idx * VT + local_value_idx] =
        state[(hv_idx * head_v_dim + value_base + local_value_idx) * HEAD_K_DIM + local_key_idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const uint token_base = chunk_idx * OUTPUT_STATE_CHUNK;
    const uint valid_tokens = token_base < suffix_len ? min(uint(OUTPUT_STATE_CHUNK), suffix_len - token_base) : 0u;
    const uint chunk_head_base = (chunk_idx * num_v_heads + hv_idx);

    TokenValueFragment output_acc;
    // -- R phase: R = beta (.) (V - e^g (.) (K . S^T)) into residual_new_value_tile ----------
    if (simdgroup_idx < NUM_TOKEN_TILES) {
      const uint token_tile_base = simdgroup_idx * TOKEN_TILE;
      const uint valid_token_rows =
          token_tile_base < valid_tokens ? min(uint(TOKEN_TILE), valid_tokens - token_tile_base) : 0u;

      TokenValueFragment residual_acc;
      residual_acc.clear();
      const device float* k_tile = k_norm + (token_base + token_tile_base) * key_dim + hk_idx * HEAD_K_DIM;
      output_acc.clear();
      const uint query_tile_base = (token_base + token_tile_base) * key_dim + hk_idx * HEAD_K_DIM;
      for (uint key_block_start = 0; key_block_start < HEAD_K_DIM; key_block_start += FR) {
        TokenKeyFragment key_frag;
        TokenKeyFragment query_frag;
        KeyValueFragment state_frag;
        key_frag.load_from(lane, fragment_source(k_tile + key_block_start, int(key_dim)).bounded(valid_token_rows, FR));
        query_frag.load_from(
            lane,
            fragment_source(q_norm + query_tile_base + key_block_start, int(key_dim)).bounded(valid_token_rows, FR)
        );
        state_frag.load_from(lane, fragment_source(transposed_state_tile + key_block_start * VT, int(VT), 1));
        fragment_mma(residual_acc, key_frag, state_frag);
        fragment_mma(output_acc, query_frag, state_frag);
      }

      const device T* value_input_tile =
          in_proj + (token_base + token_tile_base) * total_proj_dim + 2 * key_dim + hv_idx * head_v_dim + value_base;
      residual_acc.map_coords(lane, [&](short row, short col, float state_projection) {
        if (uint(row) >= valid_token_rows) {
          return 0.0f;
        }
        const uint token = token_base + token_tile_base + uint(row);
        const float beta_t = beta[token * num_v_heads + hv_idx];
        const float g_t = g_head[token];
        const float value_input = float(value_input_tile[uint(row) * total_proj_dim + uint(col)]);
        return beta_t * (value_input - fast::exp(g_t) * state_projection);
      });
      residual_acc.store(lane, residual_new_value_tile + token_tile_base * VT, int(VT));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Vnew phase: Vnew = T . R (dense bf16 T streamed from device) --------
    TokenValueFragment new_value_acc;
    if (simdgroup_idx < NUM_TOKEN_TILES) {
      const uint token_tile_base = simdgroup_idx * TOKEN_TILE;
      const uint valid_token_rows =
          token_tile_base < valid_tokens ? min(uint(TOKEN_TILE), valid_tokens - token_tile_base) : 0u;

      new_value_acc.clear();
      const device bfloat* causal_inv_rows =
          t_mat + chunk_head_base * OUTPUT_STATE_CHUNK * OUTPUT_STATE_CHUNK + token_tile_base * OUTPUT_STATE_CHUNK;
      // Skip fully upper-causal T tiles.
      const uint causal_col_end = min(uint(OUTPUT_STATE_CHUNK), token_tile_base + TOKEN_TILE);
      for (uint source_token_block_start = 0; source_token_block_start < causal_col_end;
           source_token_block_start += FR) {
        TokenKeyFragment causal_inv_frag;
        ScratchKeyValueFragment residual_frag;
        causal_inv_frag.load_from(
            lane,
            fragment_source(causal_inv_rows + source_token_block_start, int(OUTPUT_STATE_CHUNK))
                .bounded(valid_token_rows, FR)
        );
        residual_frag.load_from(
            lane,
            fragment_source(residual_new_value_tile + source_token_block_start * VT, int(VT), 1)
        );
        fragment_mma(new_value_acc, causal_inv_frag, residual_frag);
      }
    }
    // Barrier before overwriting residual_new_value_tile (R) with Vnew: all simdgroups must be
    // done reading R above.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_idx < NUM_TOKEN_TILES) {
      const uint token_tile_base = simdgroup_idx * TOKEN_TILE;
      new_value_acc.store(lane, residual_new_value_tile + token_tile_base * VT, int(VT));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Y phase: Y = e^g (.) (Q . S^T) + A . Vnew ; cast+store to out -------
    if (simdgroup_idx < NUM_TOKEN_TILES) {
      const uint token_tile_base = simdgroup_idx * TOKEN_TILE;
      const uint valid_token_rows =
          token_tile_base < valid_tokens ? min(uint(TOKEN_TILE), valid_tokens - token_tile_base) : 0u;

      output_acc.map_coords(lane, [&](short row, short, float value) {
        if (uint(row) >= valid_token_rows) {
          return 0.0f;
        }
        const uint token = token_base + token_tile_base + uint(row);
        return value * fast::exp(g_head[token]);
      });

      const uint qk_tile_base =
          chunk_head_base * OUTPUT_STATE_CHUNK * OUTPUT_STATE_CHUNK + token_tile_base * OUTPUT_STATE_CHUNK;
      // Skip fully upper-causal qk tiles.
      const uint causal_col_end = min(uint(OUTPUT_STATE_CHUNK), token_tile_base + TOKEN_TILE);
      for (uint source_token_block_start = 0; source_token_block_start < causal_col_end;
           source_token_block_start += FR) {
        const uint valid_source_tokens =
            source_token_block_start < valid_tokens ? min(uint(FR), valid_tokens - source_token_block_start) : 0u;
        TokenKeyFragment qk_frag;
        ScratchKeyValueFragment new_value_frag;
        qk_frag.load_from(
            lane,
            fragment_source(qk_scaled + qk_tile_base + source_token_block_start, int(OUTPUT_STATE_CHUNK))
                .bounded(valid_token_rows, valid_source_tokens)
        );
        new_value_frag.load_from(
            lane,
            fragment_source(residual_new_value_tile + source_token_block_start * VT, int(VT), 1)
        );
        fragment_mma(output_acc, qk_frag, new_value_frag);
      }

      const uint output_tile_base = (token_base + token_tile_base) * value_dim + hv_idx * head_v_dim + value_base;
      output_acc.store_safe(lane, out + output_tile_base, int(value_dim), short2(short(VT), short(valid_token_rows)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Update phase: S^T <- exp(g_last) . S^T + (decay_scale (.) K)^T . Vnew -----
    {
      const uint key_tile_base = simdgroup_idx * OUTPUT_STATE_KEY_TILE;
      const uint g_last_token = token_base + (valid_tokens > 0 ? valid_tokens - 1 : 0u);
      const float g_last = g_head[g_last_token];
      const float state_decay = fast::exp(g_last);

      ValueKeyFragment state_update_acc;
      state_update_acc.clear();
      for (uint source_token_block_start = 0; source_token_block_start < OUTPUT_STATE_CHUNK;
           source_token_block_start += FR) {
        const uint valid_source_tokens =
            source_token_block_start < valid_tokens ? min(uint(FR), valid_tokens - source_token_block_start) : 0u;
        ScratchValueTokenFragment new_value_frag;
        TokenKeyTileFragment key_frag;
        new_value_frag.load_from(
            lane,
            fragment_source(residual_new_value_tile + source_token_block_start * VT, 1, int(VT))
                .bounded(short(VT), short(valid_source_tokens))
        );
        const device float* k_tile =
            k_norm + (token_base + source_token_block_start) * key_dim + hk_idx * HEAD_K_DIM + key_tile_base;
        key_frag.load_from(
            lane,
            fragment_source(k_tile, int(key_dim)).bounded(short(valid_source_tokens), short(OUTPUT_STATE_KEY_TILE))
        );
        key_frag.map_coords(lane, [&](short row, short, float value) {
          if (uint(row) >= valid_source_tokens) {
            return 0.0f;
          }
          const uint token = token_base + source_token_block_start + uint(row);
          // beta is already folded into Vnew through R.
          return value * fast::exp(g_last - g_head[token]);
        });
        fragment_mma(state_update_acc, new_value_frag, key_frag);
      }

      state_update_acc.map_coords(lane, [&](short row, short col, float value) {
        return state_decay * transposed_state_tile[(key_tile_base + uint(col)) * VT + uint(row)] + value;
      });
      state_update_acc.store(lane, transposed_state_tile + key_tile_base * VT, 1, int(VT));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write the final state slice back to device, transposing [K, VT] -> [V, K].
  for (uint idx = thread_idx; idx < VT * HEAD_K_DIM; idx += OUTPUT_STATE_THREADS) {
    const uint local_value_idx = idx / HEAD_K_DIM;
    const uint local_key_idx = idx - local_value_idx * HEAD_K_DIM;
    state[(hv_idx * head_v_dim + value_base + local_value_idx) * HEAD_K_DIM + local_key_idx] =
        transposed_state_tile[local_key_idx * VT + local_value_idx];
  }
}
