#include <metal_stdlib>
#include "../definitions.metal"
#include "../matmul/common/loader.h"
#include "../matmul/common/mma.h"
#include "attention.h"

using namespace metal;
using namespace uzu::matmul;
using namespace uzu::attention;

template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T scale_) : scale(scale_) {}
  METAL_FUNC T apply(T x) const { return scale * x; }
};

template <typename T>
METAL_FUNC T row_reduce_max(T v) {
  // BaseMMAFrag::get_coord mapping groups lanes for a row as:
  // {lane, lane^1, lane^8, (lane^1)^8}. Reduce in two steps.
  v = metal::max(v, simd_shuffle_xor(v, 1));
  v = metal::max(v, simd_shuffle_xor(v, 8));
  return v;
}

template <typename T>
METAL_FUNC T row_reduce_sum(T v) {
  v += simd_shuffle_xor(v, 1);
  v += simd_shuffle_xor(v, 8);
  return v;
}

#define BQ 32
#define WM 4
#define WN 1

template <typename T, uint BK, uint BD>
VARIANTS(T, float, half, bfloat)
VARIANTS(BK, 16, 32)
VARIANTS(BD, 64, 128, 256)
KERNEL(AttentionGemm)(
    const device T* q,
    const device T* k,
    const device T* v,
    device T* o,
    const constant AttnParams& params,
    const constant AttnMaskParams& mask_params OPTIONAL(has_mask),
    const device T* mask OPTIONAL(has_mask),
    const device float* sinks OPTIONAL(has_sinks),
    const constant uint& num_heads,
    const constant uint& suffix_length,
    const bool align_q SPECIALIZE,
    const bool align_k SPECIALIZE,
    const bool do_causal SPECIALIZE,
    const bool has_mask SPECIALIZE,
    const bool has_sinks SPECIALIZE,
    threadgroup T q_smem[BQ * (BD + 16 / sizeof(T))],
    threadgroup T kv_smem[BK * (BD + 16 / sizeof(T))],
    const Simd simd,
    const uint tgid_x GROUPS(suffix_length.div_ceil(BQ)),
    const uint tgid_y GROUPS(num_heads),
    const uint tgid_z GROUPS(1),
    const uint lid THREADS(128)
) {
  // -------------------------------------------------------------------------
  // Pointer setup (all strides are in elements)
  // tgid_x: query tile index (BQ rows)
  // tgid_y: query head index
  // tgid_z: batch index (currently 1 in uzu, but kept for completeness)
  const uint batch_idx = tgid_z;
  const uint head_idx = tgid_y;
  const uint q_tile_idx = tgid_x;

  q += batch_idx * params.q_strides[0] + head_idx * params.q_strides[1] +
       q_tile_idx * int64_t(BQ) * params.q_strides[2];

  const int kv_head_idx = int(tgid_y) / params.gqa_factor;
  k += batch_idx * params.k_strides[0] +
       int64_t(kv_head_idx) * params.k_strides[1];
  v += batch_idx * params.v_strides[0] +
       int64_t(kv_head_idx) * params.v_strides[1];

  o += batch_idx * params.o_strides[0] + head_idx * params.o_strides[1] +
       q_tile_idx * int64_t(BQ) * params.o_strides[2];

  if (has_mask) {
    mask += batch_idx * mask_params.m_strides[0] +
            head_idx * mask_params.m_strides[1];
  }

  // -------------------------------------------------------------------------
  // Threadgroup memory
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BD + padK; // K stored as [BK, BD] row-major
  constexpr short LDV_tgp = BD + padV;

  threadgroup T* Qs = q_smem;
  threadgroup T* Ks = kv_smem;
  threadgroup T* Vs = kv_smem;

  //
  // -------------------------------------------------------------------------
  // Block loaders
  using QBlockLoader = BlockLoader<
      T,
      /*BROWS=*/BQ,
      /*BCOLS=*/BD,
      /*dst_ld=*/LDQ_tgp,
      /*reduction_dim=*/1,
      /*tgp_size=*/WM * WN * 32>;

  using KBlockLoader = BlockLoader<
      T,
      /*BROWS=*/BK,
      /*BCOLS=*/BD,
      /*dst_ld=*/LDK_tgp,
      /*reduction_dim=*/0,
      /*tgp_size=*/WM * WN * 32>;

  using VBlockLoader = BlockLoader<
      T,
      /*BROWS=*/BK,
      /*BCOLS=*/BD,
      /*dst_ld=*/LDV_tgp,
      /*reduction_dim=*/0,
      /*tgp_size=*/WM * WN * 32>;

  const int q_src_ld = int(params.q_strides[2]);
  const int k_src_ld = int(params.k_strides[2]);
  const int v_src_ld = int(params.v_strides[2]);

  thread QBlockLoader loader_q(q, q_src_ld, Qs, simd.group_idx, simd.lane_idx);
  thread KBlockLoader loader_k(k, k_src_ld, Ks, simd.group_idx, simd.lane_idx);
  thread VBlockLoader loader_v(v, v_src_ld, Vs, simd.group_idx, simd.lane_idx);

  TransformScale<T> ts(static_cast<T>(params.scale * M_LOG2E_F));

  // -------------------------------------------------------------------------
  // MMA tiles
  constexpr short kFragSize = 8;
  using AccumType = float;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host at least 1 simdgroup matrix along Q sequence."
  );

  // Q seq frags per warp (we keep TQ == 1 for the 32-row block layout)
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  // KV sequence frags
  constexpr int TK = BK / kFragSize;
  // HeadDim frags
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Expected TQ == 1");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile; // represents K^T slice
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  // -------------------------------------------------------------------------
  // Lane coordinates and pointer offsets
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd.lane_idx);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  const short tm = kFragSize * TQ * short(simd.group_idx);

  // Qs is row-major [BQ, BD]
  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  constexpr short Qs_tile_stride = kFragSize;

  // Ks is row-major [BK, BD] but we load K^T by swapping strides:
  // B_str_k = 1, B_str_n = LDK_tgp => offset = sn * LDK_tgp + sm
  const short Ks_offset = sn * LDK_tgp + sm;
  constexpr short Ks_tile_stride =
      kFragSize; // advance along head-dim (contiguous)

  // Vs is row-major [BK, BD]
  const short Vs_offset = sm * LDV_tgp + sn;

  // -------------------------------------------------------------------------
  // Load Q block once (and apply scaling)
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (!align_q && int(tgid_x) == params.nq_aligned) {
    loader_q.load_safe(short2(BD, params.q_rem));
  } else {
    loader_q.load_unsafe();
  }
  loader_q.apply_inplace_op(ts);

  // -------------------------------------------------------------------------
  // Streaming softmax state for this row (shared across lanes in a row)
  const AccumType neg_inf = static_cast<AccumType>(-1e9f) * M_LOG2E_F;
  AccumType max_score = -INFINITY;
  AccumType sum_score = AccumType(0);

  if (has_sinks) {
    max_score = M_LOG2E_F * static_cast<AccumType>(sinks[tgid_y]);
    sum_score = AccumType(1);
  }

  // Determine K block loop limit (causal can early-stop)
  int kb_lim = params.nk;
  if (do_causal) {
    const int q_max = (int(tgid_x) + 1) * BQ + params.q_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params.nk, kb_lim);
  }

  const int q_rel = int(tgid_x) * BQ + int(tm) + int(sm); // [0, q_len)
  const int q_abs = q_rel + params.q_off;                 // [0, k_len)

  // Loop over KV blocks
  for (int kb = 0; kb < kb_lim; kb++) {
    // Load K block
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_k && kb == params.nk_aligned) {
      loader_k.load_safe(short2(BD, params.k_rem));
    } else {
      loader_k.load_unsafe();
    }

    // Compute S = Q @ K^T for this block
    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    UZU_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]
      );
      Ktile.template load<T, 1, 1, 1, LDK_tgp>(
          &Ks[Ks_offset + dd * Ks_tile_stride]
      );

      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // Mask out tail keys for the last (unaligned) K block
    if (!align_k && kb == params.nk_aligned) {
      const int k_rem = params.k_rem;
      UZU_PRAGMA_UNROLL
      for (short j = 0; j < TK; j++) {
        thread auto& frag = Stile.frag_at(0, j);
        const int col0 = int(sn) + int(j) * kFragSize;
        if (col0 >= k_rem) {
          frag[0] = neg_inf;
        }
        if ((col0 + 1) >= k_rem) {
          frag[1] = neg_inf;
        }
      }
    }

    // Causal mask (only needed for the last few blocks near the diagonal)
    if (do_causal) {
      const int tail_blocks = (BQ + BK - 1) / BK + int(!align_k);
      const int tail_start = kb_lim - tail_blocks;
      if (kb >= tail_start) {
        UZU_PRAGMA_UNROLL
        for (short j = 0; j < TK; j++) {
          thread auto& frag = Stile.frag_at(0, j);
          const int col_base = kb * BK + int(sn) + int(j) * kFragSize;
          if (q_abs < col_base) {
            frag[0] = neg_inf;
          }
          if (q_abs < (col_base + 1)) {
            frag[1] = neg_inf;
          }
        }
      }
    }

    // Add external mask (additive bias in natural-log domain; convert to log2)
    if (has_mask && q_rel < params.q_len) {
      const int64_t row_stride = mask_params.m_strides[2];
      const int64_t row_base = int64_t(q_rel) * row_stride;

      UZU_PRAGMA_UNROLL
      for (short j = 0; j < TK; j++) {
        thread auto& frag = Stile.frag_at(0, j);
        const int col_base = kb * BK + int(sn) + int(j) * kFragSize;

        const int k0 = col_base;
        const int k1 = col_base + 1;

        if (k0 < params.k_len) {
          AccumType mv = static_cast<AccumType>(mask[row_base + int64_t(k0)]);
          mv = metal::max(mv, static_cast<AccumType>(-1e9f));
          frag[0] += M_LOG2E_F * mv;
        }
        if (k1 < params.k_len) {
          AccumType mv = static_cast<AccumType>(mask[row_base + int64_t(k1)]);
          mv = metal::max(mv, static_cast<AccumType>(-1e9f));
          frag[1] += M_LOG2E_F * mv;
        }
      }
    }

    // Load V block (overwriting K in shared memory)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_k && kb == params.nk_aligned) {
      loader_v.load_safe(short2(BD, params.k_rem));
    } else {
      loader_v.load_unsafe();
    }

    // -----------------------------------------------------------------------
    // Streaming softmax update for this block

    // Row max for this block
    AccumType block_max_local = -INFINITY;
    UZU_PRAGMA_UNROLL
    for (short j = 0; j < TK; j++) {
      const thread auto& frag = Stile.frag_at(0, j);
      block_max_local = metal::max(block_max_local, frag[0]);
      block_max_local = metal::max(block_max_local, frag[1]);
    }
    const AccumType block_max = row_reduce_max(block_max_local);

    const AccumType new_max = metal::max(max_score, block_max);
    const AccumType factor = fast::exp2(max_score - new_max);
    max_score = new_max;

    // Rescale running sum
    sum_score *= factor;

    // exp2(S - new_max) and row sum
    AccumType block_sum_local = AccumType(0);
    UZU_PRAGMA_UNROLL
    for (short j = 0; j < TK; j++) {
      thread auto& frag = Stile.frag_at(0, j);
      frag[0] = fast::exp2(frag[0] - new_max);
      frag[1] = fast::exp2(frag[1] - new_max);
      block_sum_local += frag[0] + frag[1];
    }
    const AccumType block_sum = row_reduce_sum(block_sum_local);
    sum_score += block_sum;

    // Rescale output accumulator
    UZU_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      thread auto& frag = Otile.frag_at(0, id);
      frag[0] *= factor;
      frag[1] *= factor;
    }

    // Accumulate output: Otile += Stile * Vblock
    threadgroup_barrier(mem_flags::mem_threadgroup);
    UZU_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      UZU_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        IF_CONSTEXPR(BD == 128) { simdgroup_barrier(mem_flags::mem_none); }

        const short kk = ik * kFragSize;
        const short dd = id * kFragSize;

        Vtile.template load<T, 1, 1, LDV_tgp, 1>(
            &Vs[Vs_offset + kk * LDV_tgp + dd]
        );

        IF_CONSTEXPR(BD == 128) { simdgroup_barrier(mem_flags::mem_none); }

        MMAFrag_acc_t::mma(
            Otile.frag_at(0, id),
            Stile.frag_at(0, ik),
            Vtile.frag_at(0, 0),
            Otile.frag_at(0, id)
        );
      }
    }

    // Prepare for next iteration
    loader_k.next();
    loader_v.next();
  }

  // -------------------------------------------------------------------------
  // Normalize output by sum_score (avoid div-by-zero for masked-out rows)
  const AccumType inv_sum = AccumType(1) / sum_score;
  UZU_PRAGMA_UNROLL
  for (short id = 0; id < TD; id++) {
    thread auto& frag = Otile.frag_at(0, id);
    frag[0] *= inv_sum;
    frag[1] *= inv_sum;
  }

  threadgroup_barrier(mem_flags::mem_none);

  // Store results (O is row-major with row-stride params.o_strides[2])
  o += int64_t(tm + sm) * params.o_strides[2] + int64_t(sn);

  if (!align_q && int(tgid_x) == params.nq_aligned) {
    const short2 dst_tile_dims = short2(BD - sn, params.q_rem - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) {
      return;
    }

    Otile.template store_safe<T, 1, 1>(
        o,
        int(params.o_strides[2]),
        dst_tile_dims
    );
  } else {
    Otile.template store<T, 1, 1>(o, int(params.o_strides[2]));
  }
}
