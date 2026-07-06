#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/mxu_fragment_ops.h"
#include "../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;

// Persistent chunk-scan kernel: one threadgroup owns one (head, value-slice)
// and marches through every chunk with state living entirely in threadgroup
// memory (S^T [K, VT] + Vnew [C, VT]). Replaces the serial
// Vnew -> UpdateDecayScale chain plus the chunk-parallel OutputA, eliminating
// the h / v_new device temporaries. State is f32 end-to-end; W/U stay bf16.
#define FUSED_THREADS 128
#define FUSED_NUM_SIMDGROUPS (FUSED_THREADS / METAL_SIMD_SIZE)
#define FUSED_HEAD_K_DIM 128
#define FUSED_CHUNK 64
// Update splits K=128 across the 4 simdgroups (32 key columns each).
#define FUSED_KEY_TILE (FUSED_HEAD_K_DIM / FUSED_NUM_SIMDGROUPS)

template <typename T, uint VT>
VARIANTS(T, float, half, bfloat)
VARIANTS(VT, 16, 32)
PUBLIC KERNEL(DeltaNetChunkedFusedApply)(
    device const bfloat* w,
    device const bfloat* u,
    device const float* q_norm,
    device const float* k_norm,
    device const float* qk_scaled,
    device const float* g,
    device const float* decay_scale,
    device float* state,
    device T* out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const uint& suffix_len,
    threadgroup float st[FUSED_HEAD_K_DIM * VT],
    threadgroup float vnew[FUSED_CHUNK * VT],
    const ThreadContext thread_context,
    const uint hv_idx GROUPS(num_v_heads),
    const uint v_slice GROUPS(head_v_dim.div_ceil(VT)),
    const uint tid THREADS(FUSED_THREADS)
) {
  // VT == 32 supplies two 16-wide value column fragments, so the MXU (16x16
  // fragments, MMA paired into 16x32) is a clean fit and each simdgroup owns a
  // 16-token tile (4 tiles / chunk). VT == 16 is a single 16-wide value column;
  // the MXU cannot express that as an N==1 output (its MMA computes one left x
  // two rights, so a value-only column would require the unsupported one-right
  // x two-lefts / token-pairing shape). Use the simdgroup 8x8 path there, which
  // has no pairing constraint. State/Vnew stay f32 either way.
  using Ops = metal::conditional_t<(VT >= 32), MxuFragmentOps<>, SimdgroupFragmentOps>;
  constexpr ushort FR = Ops::FRAGMENT_ROWS; // 16 (MXU) or 8 (simdgroup)
  constexpr ushort FC = Ops::FRAGMENT_COLS;
  static_assert(FR == FC, "fused kernel assumes square fragments");
  static_assert(FUSED_HEAD_K_DIM % FR == 0, "K must tile the fragment rows");
  static_assert(FUSED_CHUNK % FR == 0, "chunk size must tile the fragment rows");
  static_assert(VT % FC == 0, "value slice must tile the fragment columns");
  static_assert(FUSED_KEY_TILE % FC == 0, "key tile must tile the fragment columns");

  // Each active simdgroup owns a token tile. VT == 32 uses 16-token tiles (4
  // tiles / chunk, all simdgroups busy). VT == 16 uses 32-token tiles (2 tiles /
  // chunk, 2 idle simdgroups in the Vnew/Y phases); the value dimension still
  // supplies >= 2 simdgroup column fragments so the MMA is well-formed.
  constexpr uint TOKEN_TILE = (VT >= 32) ? 16u : 32u;
  constexpr uint NUM_TOKEN_TILES = FUSED_CHUNK / TOKEN_TILE;
  constexpr ushort TOKEN_FRAGMENTS = TOKEN_TILE / FR;
  constexpr ushort VALUE_FRAGMENTS = VT / FC;
  constexpr ushort KEY_FRAGMENTS = FUSED_KEY_TILE / FC;

  using YAccFragment = Fragment<float, TOKEN_FRAGMENTS, VALUE_FRAGMENTS, Ops>;
  using LeftKFragment = OperandFragment<float, TOKEN_FRAGMENTS, 1, Ops>; // [tokens, k/j]
  using StateFragment = OperandFragment<float, 1, VALUE_FRAGMENTS, Ops>; // [k/j, value]
  using UpdAccFragment = Fragment<float, VALUE_FRAGMENTS, KEY_FRAGMENTS, Ops>;
  using VnewColFragment = OperandFragment<float, VALUE_FRAGMENTS, 1, Ops>; // [value, j]
  using KeyRowFragment = OperandFragment<float, 1, KEY_FRAGMENTS, Ops>;    // [j, key]

  const uint lane = thread_context.simd_lane_id;
  const uint sg = thread_context.simdgroup_index;
  const uint value_base = v_slice * VT;
  if (value_base >= head_v_dim) {
    return;
  }
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk_idx = hv_idx / groups_per_head;
  const uint num_chunks = (suffix_len + FUSED_CHUNK - 1) / FUSED_CHUNK;

  // Load the initial state slice transposed into threadgroup memory:
  //   st[k * VT + v] = state[(hv, value_base + v, k)]
  for (uint idx = tid; idx < VT * FUSED_HEAD_K_DIM; idx += FUSED_THREADS) {
    const uint v = idx / FUSED_HEAD_K_DIM;
    const uint k = idx - v * FUSED_HEAD_K_DIM;
    st[k * VT + v] = state[(hv_idx * head_v_dim + value_base + v) * FUSED_HEAD_K_DIM + k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const uint token_base = chunk_idx * FUSED_CHUNK;
    const uint valid_tokens = token_base < suffix_len ? min(uint(FUSED_CHUNK), suffix_len - token_base) : 0u;
    const uint chunk_head_base = (chunk_idx * num_v_heads + hv_idx);

    // -- Phase 1: Vnew = U - W . S^T  (streamed W bf16, S^T from TG) ----------
    if (sg < NUM_TOKEN_TILES) {
      const uint row_base = sg * TOKEN_TILE;
      const uint valid_rows = row_base < valid_tokens ? min(uint(TOKEN_TILE), valid_tokens - row_base) : 0u;

      YAccFragment acc;
      acc.clear();
      const device bfloat* w_head = w + (chunk_head_base * FUSED_CHUNK + row_base) * FUSED_HEAD_K_DIM;
      for (uint k0 = 0; k0 < FUSED_HEAD_K_DIM; k0 += FR) {
        LeftKFragment w_frag;
        StateFragment s_frag;
        w_frag.load_from(lane, fragment_source(w_head + k0, int(FUSED_HEAD_K_DIM)).bounded(valid_rows, FR));
        s_frag.load_from(lane, fragment_source(st + k0 * VT, int(VT), 1));
        fragment_mma(acc, w_frag, s_frag);
      }

      const device bfloat* u_tile = u + (chunk_head_base * FUSED_CHUNK + row_base) * head_v_dim + value_base;
      acc.map_coords(lane, [&](short row, short col, float correction) {
        if (uint(row) >= valid_rows) {
          return 0.0f;
        }
        return float(u_tile[uint(row) * head_v_dim + uint(col)]) - correction;
      });
      acc.store(lane, vnew + row_base * VT, int(VT));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Phase 2: Y = exp(g) (.) (Q . S^T) + A . Vnew ; cast+store to out -----
    if (sg < NUM_TOKEN_TILES) {
      const uint row_base = sg * TOKEN_TILE;
      const uint valid_rows = row_base < valid_tokens ? min(uint(TOKEN_TILE), valid_tokens - row_base) : 0u;

      YAccFragment acc;
      acc.clear();
      const uint q_base = (token_base + row_base) * key_dim + hk_idx * FUSED_HEAD_K_DIM;
      for (uint k0 = 0; k0 < FUSED_HEAD_K_DIM; k0 += FR) {
        LeftKFragment q_frag;
        StateFragment s_frag;
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

      const uint qk_base = chunk_head_base * FUSED_CHUNK * FUSED_CHUNK + row_base * FUSED_CHUNK;
      for (uint j0 = 0; j0 < FUSED_CHUNK; j0 += FR) {
        const uint valid_j = j0 < valid_tokens ? min(uint(FR), valid_tokens - j0) : 0u;
        LeftKFragment a_frag;
        StateFragment v_frag;
        a_frag.load_from(
            lane,
            fragment_source(qk_scaled + qk_base + j0, int(FUSED_CHUNK)).bounded(valid_rows, valid_j)
        );
        v_frag.load_from(lane, fragment_source(vnew + j0 * VT, int(VT), 1));
        fragment_mma(acc, a_frag, v_frag);
      }

      const uint out_base = (token_base + row_base) * value_dim + hv_idx * head_v_dim + value_base;
      acc.store_safe(lane, out + out_base, int(value_dim), short2(short(VT), short(valid_rows)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Phase 3: S^T <- alpha . S^T + (decay_scale (.) K)^T . Vnew ----------
    // Each simdgroup owns one 32-key-column block of the [VT, K] update.
    {
      const uint key_base = sg * FUSED_KEY_TILE;
      const uint g_last_token = token_base + (valid_tokens > 0 ? valid_tokens - 1 : 0u);
      const float alpha = fast::exp(g[g_last_token * num_v_heads + hv_idx]);

      UpdAccFragment acc;
      acc.clear();
      for (uint j0 = 0; j0 < FUSED_CHUNK; j0 += FR) {
        const uint valid_j = j0 < valid_tokens ? min(uint(FR), valid_tokens - j0) : 0u;
        VnewColFragment v_frag;
        KeyRowFragment k_frag;
        v_frag.load_from(lane, fragment_source(vnew + j0 * VT, 1, int(VT)).bounded(short(VT), short(valid_j)));
        const device float* k_tile = k_norm + (token_base + j0) * key_dim + hk_idx * FUSED_HEAD_K_DIM + key_base;
        k_frag.load_from(lane, fragment_source(k_tile, int(key_dim)).bounded(short(valid_j), short(FUSED_KEY_TILE)));
        k_frag.map_coords(lane, [&](short row, short, float value) {
          if (uint(row) >= valid_j) {
            return 0.0f;
          }
          const uint local_t = j0 + uint(row);
          return value * decay_scale[chunk_head_base * FUSED_CHUNK + local_t];
        });
        fragment_mma(acc, v_frag, k_frag);
      }

      // acc holds Vnew^T . B for value rows [0, VT) and key cols [key_base, +32).
      // Fold in alpha . S^T (read transposed) and store back transposed. Each
      // simdgroup owns a disjoint key block, so no intra-phase barrier is
      // needed between the read (map_coords) and the write (store).
      acc.map_coords(lane, [&](short row, short col, float value) {
        return alpha * st[(key_base + uint(col)) * VT + uint(row)] + value;
      });
      acc.store(lane, st + key_base * VT, 1, int(VT));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write the final state slice back to device, transposing [K, VT] -> [V, K].
  for (uint idx = tid; idx < VT * FUSED_HEAD_K_DIM; idx += FUSED_THREADS) {
    const uint v = idx / FUSED_HEAD_K_DIM;
    const uint k = idx - v * FUSED_HEAD_K_DIM;
    state[(hv_idx * head_v_dim + value_base + v) * FUSED_HEAD_K_DIM + k] = st[k * VT + v];
  }
}
