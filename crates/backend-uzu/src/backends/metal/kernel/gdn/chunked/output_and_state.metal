#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../common/thread_context.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/mxu_fragment_ops.h"
#include "../../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;

// Chunked Mode-L scan. T is the dense chunk inverse from DenseCausalInverse;
// scratch holds R and then Vnew. State and qk_scaled stay f32; dense T is bf16.
#define OUTPUT_STATE_THREADS 128
#define OUTPUT_STATE_NUM_SIMDGROUPS (OUTPUT_STATE_THREADS / METAL_SIMD_SIZE)
#define OUTPUT_STATE_HEAD_K_DIM 128
#define OUTPUT_STATE_CHUNK 64
#define OUTPUT_STATE_KEY_TILE (OUTPUT_STATE_HEAD_K_DIM / OUTPUT_STATE_NUM_SIMDGROUPS)

// USE_MXU selects the fragment backend. Both paths accumulate in f32; only the
// hardware fragment shape changes.
template <typename T, typename O, uint VT, bool USE_MXU>
VARIANTS(T, float, bfloat)
VARIANTS(O, float, bfloat)
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
    threadgroup float st[OUTPUT_STATE_HEAD_K_DIM * VT],
    threadgroup float scratch[OUTPUT_STATE_CHUNK * VT],
    const ThreadContext thread_context,
    const uint hv_idx GROUPS(num_v_heads),
    const uint v_slice GROUPS(head_v_dim.div_ceil(VT)),
    const uint tid THREADS(OUTPUT_STATE_THREADS)
) {
  using Ops = metal::conditional_t<USE_MXU, MxuFragmentOps<>, SimdgroupFragmentOps>;
  constexpr ushort FR = Ops::FRAGMENT_ROWS;
  constexpr ushort FC = Ops::FRAGMENT_COLS;
  static_assert(FR == FC, "output-state kernel assumes square fragments");
  static_assert(OUTPUT_STATE_HEAD_K_DIM % FR == 0, "K must tile the fragment rows");
  static_assert(OUTPUT_STATE_CHUNK % FR == 0, "chunk size must tile the fragment rows");
  static_assert(VT % FC == 0, "value slice must tile the fragment columns");
  static_assert(OUTPUT_STATE_KEY_TILE % FC == 0, "key tile must tile the fragment columns");

  constexpr uint TOKEN_TILE = (VT >= 32) ? 16u : 32u;
  constexpr uint NUM_TOKEN_TILES = OUTPUT_STATE_CHUNK / TOKEN_TILE;
  constexpr ushort TOKEN_FRAGMENTS = TOKEN_TILE / FR;
  constexpr ushort VALUE_FRAGMENTS = VT / FC;
  constexpr ushort KEY_FRAGMENTS = OUTPUT_STATE_KEY_TILE / FC;

  using AccFragment = Fragment<float, TOKEN_FRAGMENTS, VALUE_FRAGMENTS, Ops>; // [tokens, value]
  using LeftFragment = OperandFragment<float, TOKEN_FRAGMENTS, 1, Ops>;       // [tokens, k/j]
  using RightFragment = OperandFragment<float, 1, VALUE_FRAGMENTS, Ops>;      // [k/j, value]
  using UpdAccFragment = Fragment<float, VALUE_FRAGMENTS, KEY_FRAGMENTS, Ops>;
  using VnewColFragment = OperandFragment<float, VALUE_FRAGMENTS, 1, Ops>; // [value, j]
  using KeyRowFragment = OperandFragment<float, 1, KEY_FRAGMENTS, Ops>;    // [j, key]

  const uint lane = thread_context.simd_lane_id;
  const uint sg = thread_context.simdgroup_index;
  const uint value_base = v_slice * VT;
  if (value_base >= head_v_dim) {
    return;
  }
  const uint hk_idx = hv_idx / (num_v_heads / num_k_heads);
  const uint num_chunks = (suffix_len + OUTPUT_STATE_CHUNK - 1) / OUTPUT_STATE_CHUNK;
  const uint conv_dim = 2 * key_dim + value_dim;
  const uint total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;

  // Load the initial state slice transposed into threadgroup memory:
  //   st[k * VT + v] = state[(hv, value_base + v, k)]
  for (uint idx = tid; idx < VT * OUTPUT_STATE_HEAD_K_DIM; idx += OUTPUT_STATE_THREADS) {
    const uint v = idx / OUTPUT_STATE_HEAD_K_DIM;
    const uint k = idx - v * OUTPUT_STATE_HEAD_K_DIM;
    st[k * VT + v] = state[(hv_idx * head_v_dim + value_base + v) * OUTPUT_STATE_HEAD_K_DIM + k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const uint token_base = chunk_idx * OUTPUT_STATE_CHUNK;
    const uint valid_tokens = token_base < suffix_len ? min(uint(OUTPUT_STATE_CHUNK), suffix_len - token_base) : 0u;
    const uint chunk_head_base = (chunk_idx * num_v_heads + hv_idx);

    // -- R phase: R = beta (.) (V - e^g (.) (K . S^T)) into scratch ----------
    if (sg < NUM_TOKEN_TILES) {
      const uint row_base = sg * TOKEN_TILE;
      const uint valid_rows = row_base < valid_tokens ? min(uint(TOKEN_TILE), valid_tokens - row_base) : 0u;

      AccFragment acc;
      acc.clear();
      const device float* k_head = k_norm + (token_base + row_base) * key_dim + hk_idx * OUTPUT_STATE_HEAD_K_DIM;
      for (uint k0 = 0; k0 < OUTPUT_STATE_HEAD_K_DIM; k0 += FR) {
        LeftFragment k_frag;
        RightFragment s_frag;
        k_frag.load_from(lane, fragment_source(k_head + k0, int(key_dim)).bounded(valid_rows, FR));
        s_frag.load_from(lane, fragment_source(st + k0 * VT, int(VT), 1));
        fragment_mma(acc, k_frag, s_frag);
      }

      const device T* v_tile =
          in_proj + (token_base + row_base) * total_proj_dim + 2 * key_dim + hv_idx * head_v_dim + value_base;
      acc.map_coords(lane, [&](short row, short col, float correction) {
        if (uint(row) >= valid_rows) {
          return 0.0f;
        }
        const uint token = token_base + row_base + uint(row);
        const float beta_t = beta[token * num_v_heads + hv_idx];
        const float g_t = g[token * num_v_heads + hv_idx];
        const float v = float(v_tile[uint(row) * total_proj_dim + uint(col)]);
        return beta_t * (v - fast::exp(g_t) * correction);
      });
      acc.store(lane, scratch + row_base * VT, int(VT));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Vnew phase: Vnew = T . R (dense bf16 T streamed from device) --------
    AccFragment vnew_acc;
    if (sg < NUM_TOKEN_TILES) {
      const uint row_base = sg * TOKEN_TILE;
      const uint valid_rows = row_base < valid_tokens ? min(uint(TOKEN_TILE), valid_tokens - row_base) : 0u;

      vnew_acc.clear();
      const device bfloat* t_head =
          t_mat + chunk_head_base * OUTPUT_STATE_CHUNK * OUTPUT_STATE_CHUNK + row_base * OUTPUT_STATE_CHUNK;
      for (uint j0 = 0; j0 < OUTPUT_STATE_CHUNK; j0 += FR) {
        LeftFragment t_frag;
        RightFragment r_frag;
        t_frag.load_from(lane, fragment_source(t_head + j0, int(OUTPUT_STATE_CHUNK)).bounded(valid_rows, FR));
        r_frag.load_from(lane, fragment_source(scratch + j0 * VT, int(VT), 1));
        fragment_mma(vnew_acc, t_frag, r_frag);
      }
    }
    // Barrier before overwriting scratch (R) with Vnew: all simdgroups must be
    // done reading R above.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < NUM_TOKEN_TILES) {
      const uint row_base = sg * TOKEN_TILE;
      vnew_acc.store(lane, scratch + row_base * VT, int(VT));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Y phase: Y = e^g (.) (Q . S^T) + A . Vnew ; cast+store to out -------
    if (sg < NUM_TOKEN_TILES) {
      const uint row_base = sg * TOKEN_TILE;
      const uint valid_rows = row_base < valid_tokens ? min(uint(TOKEN_TILE), valid_tokens - row_base) : 0u;

      AccFragment acc;
      acc.clear();
      const uint q_base = (token_base + row_base) * key_dim + hk_idx * OUTPUT_STATE_HEAD_K_DIM;
      for (uint k0 = 0; k0 < OUTPUT_STATE_HEAD_K_DIM; k0 += FR) {
        LeftFragment q_frag;
        RightFragment s_frag;
        q_frag.load_from(lane, fragment_source(q_norm + q_base + k0, int(key_dim)).bounded(valid_rows, FR));
        s_frag.load_from(lane, fragment_source(st + k0 * VT, int(VT), 1));
        fragment_mma(acc, q_frag, s_frag);
      }

      acc.map_coords(lane, [&](short row, short, float value) {
        if (uint(row) >= valid_rows) {
          return 0.0f;
        }
        const uint token = token_base + row_base + uint(row);
        return value * fast::exp(g[token * num_v_heads + hv_idx]);
      });

      const uint qk_base = chunk_head_base * OUTPUT_STATE_CHUNK * OUTPUT_STATE_CHUNK + row_base * OUTPUT_STATE_CHUNK;
      for (uint j0 = 0; j0 < OUTPUT_STATE_CHUNK; j0 += FR) {
        const uint valid_j = j0 < valid_tokens ? min(uint(FR), valid_tokens - j0) : 0u;
        LeftFragment a_frag;
        RightFragment v_frag;
        a_frag.load_from(
            lane,
            fragment_source(qk_scaled + qk_base + j0, int(OUTPUT_STATE_CHUNK)).bounded(valid_rows, valid_j)
        );
        v_frag.load_from(lane, fragment_source(scratch + j0 * VT, int(VT), 1));
        fragment_mma(acc, a_frag, v_frag);
      }

      const uint out_base = (token_base + row_base) * value_dim + hv_idx * head_v_dim + value_base;
      acc.store_safe(lane, out + out_base, int(value_dim), short2(short(VT), short(valid_rows)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Update phase: S^T <- alpha . S^T + (decay_scale (.) K)^T . Vnew -----
    {
      const uint key_base = sg * OUTPUT_STATE_KEY_TILE;
      const uint g_last_token = token_base + (valid_tokens > 0 ? valid_tokens - 1 : 0u);
      const float g_last = g[g_last_token * num_v_heads + hv_idx];
      const float alpha = fast::exp(g_last);

      UpdAccFragment acc;
      acc.clear();
      for (uint j0 = 0; j0 < OUTPUT_STATE_CHUNK; j0 += FR) {
        const uint valid_j = j0 < valid_tokens ? min(uint(FR), valid_tokens - j0) : 0u;
        VnewColFragment v_frag;
        KeyRowFragment k_frag;
        v_frag.load_from(lane, fragment_source(scratch + j0 * VT, 1, int(VT)).bounded(short(VT), short(valid_j)));
        const device float* k_tile = k_norm + (token_base + j0) * key_dim + hk_idx * OUTPUT_STATE_HEAD_K_DIM + key_base;
        k_frag.load_from(
            lane,
            fragment_source(k_tile, int(key_dim)).bounded(short(valid_j), short(OUTPUT_STATE_KEY_TILE))
        );
        k_frag.map_coords(lane, [&](short row, short, float value) {
          if (uint(row) >= valid_j) {
            return 0.0f;
          }
          const uint token = token_base + j0 + uint(row);
          // decay_scale[t] = exp(g_last - g_t); beta is already baked into Vnew
          // via R, so it does not appear here (matches the fused kernel).
          return value * fast::exp(g_last - g[token * num_v_heads + hv_idx]);
        });
        fragment_mma(acc, v_frag, k_frag);
      }

      acc.map_coords(lane, [&](short row, short col, float value) {
        return alpha * st[(key_base + uint(col)) * VT + uint(row)] + value;
      });
      acc.store(lane, st + key_base * VT, 1, int(VT));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write the final state slice back to device, transposing [K, VT] -> [V, K].
  for (uint idx = tid; idx < VT * OUTPUT_STATE_HEAD_K_DIM; idx += OUTPUT_STATE_THREADS) {
    const uint v = idx / OUTPUT_STATE_HEAD_K_DIM;
    const uint k = idx - v * OUTPUT_STATE_HEAD_K_DIM;
    state[(hv_idx * head_v_dim + value_base + v) * OUTPUT_STATE_HEAD_K_DIM + k] = st[k * VT + v];
  }
}
