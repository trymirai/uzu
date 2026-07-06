#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/mxu_fragment_ops.h"
#include "../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;

// Mode S mega kernel (INLINE_PRECOMPUTE): the small-T sibling of
// `DeltaNetChunkedMegaApply`. Instead of reading the causal-masked decay-scaled
// matrix A (qk_scaled) and the dense inverse T from device, each threadgroup
// computes them itself, per chunk, in threadgroup memory:
//   * gtile  = chunk-local cumsum of log_decay (per v-head)  [barrier]
//   * gram   = kk = K.K^T (-> A_strict) and qk = Q.K^T (-> QK)     [barrier]
//   * solve  = block 16x16 diagonal-block inverses (Dinv), then a block forward
//              substitution (identity RHS) -> dense T over A_strict in place
//              -- the SAME block algorithm as chunked_solve/chunked_solve_t,
//              NEVER a naive per-column O(C^3) substitution.
// This collapses the Mode L precompute chain (gram + solve + solveT) into the
// scan kernel, so the shipping pipeline is prep -> megaApply(inline) -> normGate
// (3 dispatches, dispatch parity with the recurrent path). The redundant gram +
// solve across the head's v-slice threadgroups (x8 for VT=16) is irrelevant at
// small T, which is dispatch-bound, not compute-bound.
//
// TG budget (32 KB cap, VT=16 ONLY -- VT=32 needs 42 KB and cannot fit):
//   st       [K=128, VT=16] f32   8 KB  (state, precision non-negotiable)
//   scratch  [C=64,  VT=16] f32   4 KB  (R / Vnew)
//   amat     [C=64,  C=64]  bf16  8 KB  (A_strict, overwritten by dense T)
//   qk_buf   [C=64,  C=64]  bf16  8 KB  (QK, the Y-phase A)
//   dinv     [C=64,  16]    bf16  2 KB  (4 diagonal-block inverses)
//   gtile    [C=64]         f32  256 B  (chunk-local g prefix)
//   total = 30.25 KB < 32 KB.
// Scratch matrices are bf16 (exactly the precision the old W/U / device T/A
// carried); state stays f32 end-to-end. A_strict -> T is done in place: block
// row i reads its A_ij (j<i) and the already-written T_j (j<i) into registers
// before a barrier, then all simdgroups store T_i -- no read/write race.
#define MEGA_THREADS 128
#define MEGA_NUM_SIMDGROUPS (MEGA_THREADS / METAL_SIMD_SIZE)
#define MEGA_HEAD_K_DIM 128
#define MEGA_CHUNK 64
#define MEGA_KEY_TILE (MEGA_HEAD_K_DIM / MEGA_NUM_SIMDGROUPS)
#define MEGA_SOLVE_BLOCK 16
#define MEGA_NUM_SOLVE_BLOCKS (MEGA_CHUNK / MEGA_SOLVE_BLOCK)

template <typename T, typename O, uint VT>
VARIANTS(T, float, half, bfloat)
VARIANTS(O, float, bfloat)
VARIANTS(VT, 16)
PUBLIC KERNEL(DeltaNetChunkedMegaApplyInline)(
    device const float* q_norm,
    device const float* k_norm,
    device const T* in_proj,
    device const float* log_decay,
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
    threadgroup bfloat amat[MEGA_CHUNK * MEGA_CHUNK],
    threadgroup bfloat qk_buf[MEGA_CHUNK * MEGA_CHUNK],
    threadgroup bfloat dinv[MEGA_CHUNK * MEGA_SOLVE_BLOCK],
    threadgroup float gtile[MEGA_CHUNK],
    const ThreadContext thread_context,
    const uint hv_idx GROUPS(num_v_heads),
    const uint v_slice GROUPS(head_v_dim.div_ceil(VT)),
    const uint tid THREADS(MEGA_THREADS)
) {
  // Mode S is VT=16, so Ops is always the 8x8 simdgroup path.
  using Ops = metal::conditional_t<(VT >= 32), MxuFragmentOps<>, SimdgroupFragmentOps>;
  constexpr ushort FR = Ops::FRAGMENT_ROWS;
  constexpr ushort FC = Ops::FRAGMENT_COLS;
  static_assert(FR == FC, "mega kernel assumes square fragments");
  static_assert(MEGA_HEAD_K_DIM % FR == 0, "K must tile the fragment rows");
  static_assert(MEGA_CHUNK % FR == 0, "chunk size must tile the fragment rows");
  static_assert(VT % FC == 0, "value slice must tile the fragment columns");
  static_assert(MEGA_KEY_TILE % FC == 0, "key tile must tile the fragment columns");
  static_assert(MEGA_SOLVE_BLOCK % FR == 0, "solve block must tile the fragment rows");

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

  // -- Inline-precompute fragment shapes (per-simdgroup 16-row gram strip and
  //    16x16 solve blocks). --------------------------------------------------
  constexpr ushort GRAM_ROW_FRAGS = MEGA_SOLVE_BLOCK / FR; // one 16-row strip / sg
  constexpr ushort GRAM_COL_FRAGS = MEGA_CHUNK / FC;       // full 64 columns
  using GramAcc = Fragment<float, GRAM_ROW_FRAGS, GRAM_COL_FRAGS, Ops>;
  using GramLeft = OperandFragment<float, GRAM_ROW_FRAGS, 1, Ops>;
  using GramRight = OperandFragment<float, 1, GRAM_COL_FRAGS, Ops, ReadTranspose>;

  constexpr ushort SOLVE_FRAGS = MEGA_SOLVE_BLOCK / FR; // 16x16 block = 2x2 frags
  using SolveTile = Fragment<float, SOLVE_FRAGS, SOLVE_FRAGS, Ops>;
  using SolveOperand = OperandFragment<float, SOLVE_FRAGS, SOLVE_FRAGS, Ops>;

  const uint lane = thread_context.simd_lane_id;
  const uint sg = thread_context.simdgroup_index;
  const uint value_base = v_slice * VT;
  if (value_base >= head_v_dim) {
    return;
  }
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk_idx = hv_idx / groups_per_head;
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

    // == Inline precompute ====================================================
    // -- gtile: chunk-local cumsum of log_decay (per token, this v-head). Each
    //    of the first C threads computes its own prefix (O(C) latency, once per
    //    chunk, OUTSIDE all O(C^2) loops -- not the F4 RECOMPUTE_G trap). ------
    if (tid < MEGA_CHUNK) {
      float acc = 0.0f;
      if (tid < valid_tokens) {
        for (uint i = 0; i <= tid; ++i) {
          acc += log_decay[(token_base + i) * num_v_heads + hv_idx];
        }
      }
      gtile[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- gram: sg computes a 16-row strip (rows sg*16..) x all 64 cols. kk feeds
    //    A_strict (c<r, beta/decay-scaled); qk feeds QK (c<=r, decay-scaled). --
    {
      const uint grow_base = sg * MEGA_SOLVE_BLOCK;
      const uint valid_rows = grow_base < valid_tokens ? min(uint(MEGA_SOLVE_BLOCK), valid_tokens - grow_base) : 0u;
      const device float* k_rows = k_norm + (token_base + grow_base) * key_dim + hk_idx * MEGA_HEAD_K_DIM;
      const device float* q_rows = q_norm + (token_base + grow_base) * key_dim + hk_idx * MEGA_HEAD_K_DIM;
      const device float* k_cols = k_norm + token_base * key_dim + hk_idx * MEGA_HEAD_K_DIM;

      GramAcc kk_acc;
      GramAcc qk_acc;
      kk_acc.clear();
      qk_acc.clear();
      for (uint d = 0; d < MEGA_HEAD_K_DIM; d += FR) {
        GramLeft k_left;
        GramLeft q_left;
        GramRight k_right;
        k_left.load_from(lane, fragment_source(k_rows + d, int(key_dim)).bounded(valid_rows, FR));
        q_left.load_from(lane, fragment_source(q_rows + d, int(key_dim)).bounded(valid_rows, FR));
        k_right.load_from(lane, fragment_source(k_cols + d, int(key_dim)).bounded(valid_tokens, FR));
        fragment_mma(kk_acc, k_left, k_right);
        fragment_mma(qk_acc, q_left, k_right);
      }

      kk_acc.map_coords(lane, [&](short r, short c, float value) {
        const uint gr = grow_base + uint(r);
        const uint gc = uint(c);
        if (gr >= valid_tokens || gc >= valid_tokens || gc >= gr) {
          return 0.0f;
        }
        const float beta_r = beta[(token_base + gr) * num_v_heads + hv_idx];
        return beta_r * fast::exp(gtile[gr] - gtile[gc]) * value;
      });
      kk_acc.store(lane, amat + grow_base * MEGA_CHUNK, int(MEGA_CHUNK));

      qk_acc.map_coords(lane, [&](short r, short c, float value) {
        const uint gr = grow_base + uint(r);
        const uint gc = uint(c);
        if (gr >= valid_tokens || gc >= valid_tokens || gc > gr) {
          return 0.0f;
        }
        return fast::exp(gtile[gr] - gtile[gc]) * value;
      });
      qk_acc.store(lane, qk_buf + grow_base * MEGA_CHUNK, int(MEGA_CHUNK));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- solve step 1: diagonal 16x16 block inverses Dinv_b = (I + A_bb)^{-1}.
    //    sg b inverts block b via 16-lane forward substitution (chunked_solve).
    //    Invalid rows have A_bb = 0 -> Dinv = I (identity carry). -------------
    {
      const uint b = sg; // MEGA_NUM_SIMDGROUPS == MEGA_NUM_SOLVE_BLOCKS == 4
      if (lane < MEGA_SOLVE_BLOCK) {
        const uint inv_col = lane;
        float inverse_col[MEGA_SOLVE_BLOCK];
        for (uint i = 0; i < MEGA_SOLVE_BLOCK; ++i) {
          inverse_col[i] = 0.0f;
        }
        inverse_col[inv_col] = 1.0f;
        for (uint inv_row = 0; inv_row < MEGA_SOLVE_BLOCK; ++inv_row) {
          if (inv_row > inv_col) {
            float acc = 0.0f;
            for (uint prev = 0; prev < inv_row; ++prev) {
              const float a =
                  float(amat[(b * MEGA_SOLVE_BLOCK + inv_row) * MEGA_CHUNK + (b * MEGA_SOLVE_BLOCK + prev)]);
              acc += a * inverse_col[prev];
            }
            inverse_col[inv_row] = -acc;
          }
        }
        for (uint inv_row = 0; inv_row < MEGA_SOLVE_BLOCK; ++inv_row) {
          dinv[b * (MEGA_SOLVE_BLOCK * MEGA_SOLVE_BLOCK) + inv_row * MEGA_SOLVE_BLOCK + inv_col] =
              bfloat(inverse_col[inv_row]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- solve step 2: block forward substitution, dense T over A_strict in
    //    place. Block row i (sequential); sg s owns column block s (cols s*16).
    //    acc_t = I_i - sum_{j<i} A_ij . T_j ; T_i = Dinv_i . acc_t. ----------
    for (uint i = 0; i < MEGA_NUM_SOLVE_BLOCKS; ++i) {
      const uint s = sg;
      SolveTile acc_t;
      acc_t.clear();
      acc_t.map_coords(lane, [&](short rl, short cl, float) {
        const uint gr = i * MEGA_SOLVE_BLOCK + uint(rl);
        const uint gc = s * MEGA_SOLVE_BLOCK + uint(cl);
        return gr == gc ? 1.0f : 0.0f;
      });
      for (uint j = 0; j < i; ++j) {
        SolveOperand a_frag;
        a_frag.load_from(
            lane,
            fragment_source(amat + (i * MEGA_SOLVE_BLOCK) * MEGA_CHUNK + j * MEGA_SOLVE_BLOCK, int(MEGA_CHUNK))
        );
        a_frag.map([](float v) { return -v; });
        SolveOperand t_frag;
        t_frag.load_from(
            lane,
            fragment_source(amat + (j * MEGA_SOLVE_BLOCK) * MEGA_CHUNK + s * MEGA_SOLVE_BLOCK, int(MEGA_CHUNK))
        );
        fragment_mma(acc_t, a_frag, t_frag);
      }
      SolveOperand inv_frag;
      inv_frag.load_from(
          lane,
          fragment_source(dinv + i * (MEGA_SOLVE_BLOCK * MEGA_SOLVE_BLOCK), int(MEGA_SOLVE_BLOCK))
      );
      SolveTile t_i;
      t_i.clear();
      fragment_mma(t_i, inv_frag, acc_t);
      // All simdgroups must finish reading A_ij / T_j before any store to
      // block row i overwrites them.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      t_i.store(lane, amat + (i * MEGA_SOLVE_BLOCK) * MEGA_CHUNK + s * MEGA_SOLVE_BLOCK, int(MEGA_CHUNK));
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // amat now holds the dense unit-lower-triangular inverse T for this chunk.

    // == Scan (identical to Mode L, sourcing T from amat and A from qk_buf) ===
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
        const uint local = row_base + uint(row);
        const float beta_t = beta[(token_base + local) * num_v_heads + hv_idx];
        const float g_t = gtile[local];
        const float v = float(v_tile[uint(row) * total_proj_dim + uint(col)]);
        return beta_t * (v - fast::exp(g_t) * correction);
      });
      acc.store(lane, scratch + row_base * VT, int(VT));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Vnew phase: Vnew = T . R (dense bf16 T from amat) --------------------
    AccFragment vnew_acc;
    if (sg < NUM_TOKEN_TILES) {
      const uint row_base = sg * TOKEN_TILE;
      const uint valid_rows = row_base < valid_tokens ? min(uint(TOKEN_TILE), valid_tokens - row_base) : 0u;

      vnew_acc.clear();
      const threadgroup bfloat* t_head = amat + row_base * MEGA_CHUNK;
      for (uint j0 = 0; j0 < MEGA_CHUNK; j0 += FR) {
        LeftFragment t_frag;
        RightFragment r_frag;
        t_frag.load_from(lane, fragment_source(t_head + j0, int(MEGA_CHUNK)).bounded(valid_rows, FR));
        r_frag.load_from(lane, fragment_source(scratch + j0 * VT, int(VT), 1));
        fragment_mma(vnew_acc, t_frag, r_frag);
      }
    }
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
        return value * fast::exp(gtile[row_base + uint(row)]);
      });

      const threadgroup bfloat* a_head = qk_buf + row_base * MEGA_CHUNK;
      for (uint j0 = 0; j0 < MEGA_CHUNK; j0 += FR) {
        const uint valid_j = j0 < valid_tokens ? min(uint(FR), valid_tokens - j0) : 0u;
        LeftFragment a_frag;
        RightFragment v_frag;
        a_frag.load_from(lane, fragment_source(a_head + j0, int(MEGA_CHUNK)).bounded(valid_rows, valid_j));
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
      const float g_last = gtile[valid_tokens > 0 ? valid_tokens - 1 : 0u];
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
          const uint local = j0 + uint(row);
          return value * fast::exp(g_last - gtile[local]);
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
