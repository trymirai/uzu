#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../common/thread_context.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/mxu_fragment_ops.h"
#include "../../matmul/common/simdgroup_fragment_ops.h"
#include "../common/heads.h"

using namespace metal;
using namespace uzu::matmul;

// Mode L mega kernel: the persistent chunk-scan of `DeltaNetChunkedFusedApply`
// with W/U (and BuildWU) eliminated via the identity
//   Vnew = T . R,   R = diag(beta) . (V - e^g (.) (K . S0^T)),
// where T is the dense unit-lower-triangular chunk inverse emitted by
// `DeltaNetChunkedSolveT`. The scan computes R itself from the state slice it
// already holds (K . S0^T is the same shape/cost as the old W . S0^T), then
// applies T as one dense [C,C] x [C,VT] MMA. State is f32 end-to-end; T / A are
// bf16 device operands (exactly the precision the old W/U carried).
//
// Per chunk (5 threadgroup barriers):
//   R phase:    R = beta*(V - e^g (.) (K . S^T))  -> scratch     [barrier]
//   Vnew phase: Vnew = T . R into registers      [barrier] store over scratch [barrier]
//   Y phase:    Y = e^g (.) (Q . S^T) + A . Vnew -> out          [barrier]
//   Update:     S^T <- alpha . S^T + (decay (.) K)^T . Vnew      [barrier]
// TG budget: S^T [K,VT] f32 (16KB) + one reused scratch [C,VT] f32 (8KB) = 24KB.
#define MEGA_THREADS 128
#define MEGA_NUM_SIMDGROUPS (MEGA_THREADS / METAL_SIMD_SIZE)
#define MEGA_HEAD_K_DIM 128
#define MEGA_CHUNK 64
#define MEGA_KEY_TILE (MEGA_HEAD_K_DIM / MEGA_NUM_SIMDGROUPS)

// VT is fixed at 32 (the strong-tile-width variant; VT=16 was dominated on every
// chip and removed). USE_MXU selects the matmul backend:
//   USE_MXU=true  -> MxuFragmentOps (16x16 fragments), the shipping M5 path.
//   USE_MXU=false -> SimdgroupFragmentOps (8x8 fragments), the M1-M4 path with
//                    no MXU.
// Operand fragments are f32 in both backends, so USE_MXU does not change operand
// precision; state is f32 end-to-end and T / A remain bf16 device operands
// regardless of backend.
template <typename T, typename O, uint VT, bool USE_MXU>
VARIANTS(T, float, half, bfloat)
VARIANTS(O, float, bfloat)
VARIANTS(VT, 32)
VARIANTS(USE_MXU, false, true)
PUBLIC KERNEL(DeltaNetChunkedMegaApply)(
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
    threadgroup float st[MEGA_HEAD_K_DIM * VT],
    threadgroup float scratch[MEGA_CHUNK * VT],
    const ThreadContext thread_context,
    const uint hv_idx GROUPS(num_v_heads),
    const uint v_slice GROUPS(head_v_dim.div_ceil(VT)),
    const uint tid THREADS(MEGA_THREADS)
) {
  // Matmul backend is chosen by USE_MXU, independent of VT. VT only controls the
  // value-tile width (and, via TOKEN_TILE below, the token split across the 4
  // simdgroups); it no longer implies the backend.
  using Ops = metal::conditional_t<USE_MXU, MxuFragmentOps<>, SimdgroupFragmentOps>;
  constexpr ushort FR = Ops::FRAGMENT_ROWS;
  constexpr ushort FC = Ops::FRAGMENT_COLS;
  static_assert(FR == FC, "mega kernel assumes square fragments");
  static_assert(MEGA_HEAD_K_DIM % FR == 0, "K must tile the fragment rows");
  static_assert(MEGA_CHUNK % FR == 0, "chunk size must tile the fragment rows");
  static_assert(VT % FC == 0, "value slice must tile the fragment columns");
  static_assert(MEGA_KEY_TILE % FC == 0, "key tile must tile the fragment columns");

  constexpr uint TOKEN_TILE = (VT >= 32) ? 16u : 32u;
  constexpr uint NUM_TOKEN_TILES = MEGA_CHUNK / TOKEN_TILE;
  constexpr ushort TOKEN_FRAGMENTS = TOKEN_TILE / FR;
  constexpr ushort VALUE_FRAGMENTS = VT / FC;
  constexpr ushort KEY_FRAGMENTS = MEGA_KEY_TILE / FC;

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
  const uint hk_idx = gdn_key_head_for_value_head(hv_idx, num_v_heads, num_k_heads);
  const uint num_chunks = (suffix_len + MEGA_CHUNK - 1) / MEGA_CHUNK;
  const uint conv_dim = 2 * key_dim + value_dim;
  const uint total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;

  // Load the initial state slice transposed into threadgroup memory:
  //   st[k * VT + v] = state[(hv, value_base + v, k)]
  for (uint idx = tid; idx < VT * MEGA_HEAD_K_DIM; idx += MEGA_THREADS) {
    const uint v = idx / MEGA_HEAD_K_DIM;
    const uint k = idx - v * MEGA_HEAD_K_DIM;
    st[k * VT + v] = state[(hv_idx * head_v_dim + value_base + v) * MEGA_HEAD_K_DIM + k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const uint token_base = chunk_idx * MEGA_CHUNK;
    const uint valid_tokens = token_base < suffix_len ? min(uint(MEGA_CHUNK), suffix_len - token_base) : 0u;
    const uint chunk_head_base = (chunk_idx * num_v_heads + hv_idx);

    // -- R phase: R = beta (.) (V - e^g (.) (K . S^T)) into scratch ----------
    if (sg < NUM_TOKEN_TILES) {
      const uint row_base = sg * TOKEN_TILE;
      const uint valid_rows = row_base < valid_tokens ? min(uint(TOKEN_TILE), valid_tokens - row_base) : 0u;

      AccFragment acc;
      acc.clear();
      const device float* k_head = k_norm + (token_base + row_base) * key_dim + hk_idx * MEGA_HEAD_K_DIM;
      for (uint k0 = 0; k0 < MEGA_HEAD_K_DIM; k0 += FR) {
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
      const device bfloat* t_head = t_mat + chunk_head_base * MEGA_CHUNK * MEGA_CHUNK + row_base * MEGA_CHUNK;
      for (uint j0 = 0; j0 < MEGA_CHUNK; j0 += FR) {
        LeftFragment t_frag;
        RightFragment r_frag;
        t_frag.load_from(lane, fragment_source(t_head + j0, int(MEGA_CHUNK)).bounded(valid_rows, FR));
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
      const uint q_base = (token_base + row_base) * key_dim + hk_idx * MEGA_HEAD_K_DIM;
      for (uint k0 = 0; k0 < MEGA_HEAD_K_DIM; k0 += FR) {
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

      const uint qk_base = chunk_head_base * MEGA_CHUNK * MEGA_CHUNK + row_base * MEGA_CHUNK;
      for (uint j0 = 0; j0 < MEGA_CHUNK; j0 += FR) {
        const uint valid_j = j0 < valid_tokens ? min(uint(FR), valid_tokens - j0) : 0u;
        LeftFragment a_frag;
        RightFragment v_frag;
        a_frag.load_from(lane, fragment_source(qk_scaled + qk_base + j0, int(MEGA_CHUNK)).bounded(valid_rows, valid_j));
        v_frag.load_from(lane, fragment_source(scratch + j0 * VT, int(VT), 1));
        fragment_mma(acc, a_frag, v_frag);
      }

      const uint out_base = (token_base + row_base) * value_dim + hv_idx * head_v_dim + value_base;
      acc.store_safe(lane, out + out_base, int(value_dim), short2(short(VT), short(valid_rows)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Update phase: S^T <- alpha . S^T + (decay_scale (.) K)^T . Vnew -----
    {
      const uint key_base = sg * MEGA_KEY_TILE;
      const uint g_last_token = token_base + (valid_tokens > 0 ? valid_tokens - 1 : 0u);
      const float g_last = g[g_last_token * num_v_heads + hv_idx];
      const float alpha = fast::exp(g_last);

      UpdAccFragment acc;
      acc.clear();
      for (uint j0 = 0; j0 < MEGA_CHUNK; j0 += FR) {
        const uint valid_j = j0 < valid_tokens ? min(uint(FR), valid_tokens - j0) : 0u;
        VnewColFragment v_frag;
        KeyRowFragment k_frag;
        v_frag.load_from(lane, fragment_source(scratch + j0 * VT, 1, int(VT)).bounded(short(VT), short(valid_j)));
        const device float* k_tile = k_norm + (token_base + j0) * key_dim + hk_idx * MEGA_HEAD_K_DIM + key_base;
        k_frag.load_from(lane, fragment_source(k_tile, int(key_dim)).bounded(short(valid_j), short(MEGA_KEY_TILE)));
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
  for (uint idx = tid; idx < VT * MEGA_HEAD_K_DIM; idx += MEGA_THREADS) {
    const uint v = idx / MEGA_HEAD_K_DIM;
    const uint k = idx - v * MEGA_HEAD_K_DIM;
    state[(hv_idx * head_v_dim + value_base + v) * MEGA_HEAD_K_DIM + k] = st[k * VT + v];
  }
}
