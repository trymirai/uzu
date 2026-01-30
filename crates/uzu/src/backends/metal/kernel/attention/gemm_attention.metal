#include <metal_stdlib>

#include "../matmul/common/loader.h"
#include "../matmul/common/mma.h"
#include "../definitions.metal"
#include "gemm_types.h"

using namespace metal;
using namespace uzu::matmul;
using namespace uzu::attention;

///////////////////////////////////////////////////////////////////////////////
// Function constants for compile-time specialization
///////////////////////////////////////////////////////////////////////////////

// Alignment flags (match Gemm convention / our matmul indices)
constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

// Feature flags
constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];

///////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// Gemm attention kernel
///////////////////////////////////////////////////////////////////////////////

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_gemm(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device T* mask [[buffer(6), function_constant(has_mask)]],
    const device float* sinks [[buffer(7), function_constant(has_sinks)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;

  // -------------------------------------------------------------------------
  // Pointer setup (all strides are in elements)
  // tid.x: query tile index (BQ rows)
  // tid.y: query head index
  // tid.z: batch index (currently 1 in uzu, but kept for completeness)
  const int64_t batch_idx = int64_t(tid.z);
  const int64_t head_idx = int64_t(tid.y);
  const int64_t q_tile_idx = int64_t(tid.x);

  Q += batch_idx * params->q_strides[0] + head_idx * params->q_strides[1] +
       q_tile_idx * int64_t(BQ) * params->q_strides[2];

  const int kv_head_idx = int(tid.y) / params->gqa_factor;
  K += batch_idx * params->k_strides[0] +
       int64_t(kv_head_idx) * params->k_strides[1];
  V += batch_idx * params->v_strides[0] +
       int64_t(kv_head_idx) * params->v_strides[1];

  O += batch_idx * params->o_strides[0] + head_idx * params->o_strides[1] +
       q_tile_idx * int64_t(BQ) * params->o_strides[2];

  if (has_mask) {
    mask += batch_idx * mask_params->m_strides[0] +
            head_idx * mask_params->m_strides[1];
  }

  // -------------------------------------------------------------------------
  // Threadgroup memory
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BD + padK; // K stored as [BK, BD] row-major
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_k = BK * (BD + padK);
  constexpr short tgp_mem_v = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_k > tgp_mem_v ? tgp_mem_k : tgp_mem_v;

  threadgroup T Q_smem[BQ * (BD + padQ)];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

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

  const int q_src_ld = int(params->q_strides[2]);
  const int k_src_ld = int(params->k_strides[2]);
  const int v_src_ld = int(params->v_strides[2]);

  thread QBlockLoader loader_q(Q, q_src_ld, Qs, simd_group_id, simd_lane_id);
  thread KBlockLoader loader_k(K, k_src_ld, Ks, simd_group_id, simd_lane_id);
  thread VBlockLoader loader_v(V, v_src_ld, Vs, simd_group_id, simd_lane_id);

  TransformScale<T> ts(static_cast<T>(params->scale * M_LOG2E_F));

  // -------------------------------------------------------------------------
  // MMA tiles
  constexpr short kFragSize = 8;
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
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  const short tm = kFragSize * TQ * short(simd_group_id);

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

  if (!align_Q && int(tid.x) == params->nq_aligned) {
    loader_q.load_safe(short2(BD, params->q_rem));
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
    max_score = M_LOG2E_F * static_cast<AccumType>(sinks[tid.y]);
    sum_score = AccumType(1);
  }

  // Determine K block loop limit (causal can early-stop)
  int kb_lim = params->nk;
  if (do_causal) {
    const int q_max = (int(tid.x) + 1) * BQ + params->q_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->nk, kb_lim);
  }

  const int q_rel = int(tid.x) * BQ + int(tm) + int(sm); // [0, q_len)
  const int q_abs = q_rel + params->q_off;               // [0, k_len)

  // Loop over KV blocks
  for (int kb = 0; kb < kb_lim; kb++) {
    // Load K block
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K && kb == params->nk_aligned) {
      loader_k.load_safe(short2(BD, params->k_rem));
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
    if (!align_K && kb == params->nk_aligned) {
      const int k_rem = params->k_rem;
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
      const int tail_blocks = (BQ + BK - 1) / BK + int(!align_K);
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
    if (has_mask && q_rel < params->q_len) {
      const int64_t row_stride = mask_params->m_strides[2];
      const int64_t row_base = int64_t(q_rel) * row_stride;

      UZU_PRAGMA_UNROLL
      for (short j = 0; j < TK; j++) {
        thread auto& frag = Stile.frag_at(0, j);
        const int col_base = kb * BK + int(sn) + int(j) * kFragSize;

        const int k0 = col_base;
        const int k1 = col_base + 1;

        if (k0 < params->k_len) {
          AccumType mv = static_cast<AccumType>(mask[row_base + int64_t(k0)]);
          mv = metal::max(mv, static_cast<AccumType>(-1e9f));
          frag[0] += M_LOG2E_F * mv;
        }
        if (k1 < params->k_len) {
          AccumType mv = static_cast<AccumType>(mask[row_base + int64_t(k1)]);
          mv = metal::max(mv, static_cast<AccumType>(-1e9f));
          frag[1] += M_LOG2E_F * mv;
        }
      }
    }

    // Load V block (overwriting K in shared memory)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K && kb == params->nk_aligned) {
      loader_v.load_safe(short2(BD, params->k_rem));
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
        IF_CONSTEXPR(BD == 128) {
          simdgroup_barrier(mem_flags::mem_none);
        }

        const short kk = ik * kFragSize;
        const short dd = id * kFragSize;

        Vtile.template load<T, 1, 1, LDV_tgp, 1>(
            &Vs[Vs_offset + kk * LDV_tgp + dd]
        );

        IF_CONSTEXPR(BD == 128) {
          simdgroup_barrier(mem_flags::mem_none);
        }

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

  // Store results (O is row-major with row-stride params->o_strides[2])
  O += int64_t(tm + sm) * params->o_strides[2] + int64_t(sn);

  if (!align_Q && int(tid.x) == params->nq_aligned) {
    const short2 dst_tile_dims = short2(BD - sn, params->q_rem - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) {
      return;
    }

    Otile.template store_safe<T, 1, 1>(
        O,
        int(params->o_strides[2]),
        dst_tile_dims
    );
  } else {
    Otile.template store<T, 1, 1>(O, int(params->o_strides[2]));
  }
}

///////////////////////////////////////////////////////////////////////////////
// Kernel instantiations (BQ=32, BK in {32,16}, WM=2, WN=2)
///////////////////////////////////////////////////////////////////////////////

#define instantiate_attention_gemm(                                            \
    type_name,                                                                 \
    element_type,                                                              \
    head_dim_value,                                                            \
    bk_value                                                                   \
)                                                                              \
  template [[host_name(                                                        \
      "attention_gemm_" #type_name "_" #head_dim_value "_bk" #bk_value         \
  )]] [[kernel]] void                                                          \
  attention_gemm<element_type, 32, bk_value, head_dim_value, 4, 1, float>(     \
      const device element_type* Q [[buffer(0)]],                              \
      const device element_type* K [[buffer(1)]],                              \
      const device element_type* V [[buffer(2)]],                              \
      device element_type* O [[buffer(3)]],                                    \
      const constant AttnParams* params [[buffer(4)]],                         \
      const constant AttnMaskParams* mask_params                               \
      [[buffer(5), function_constant(has_mask)]],                              \
      const device element_type* mask                                          \
      [[buffer(6), function_constant(has_mask)]],                              \
      const device float* sinks [[buffer(7), function_constant(has_sinks)]],   \
      uint simd_lane_id [[thread_index_in_simdgroup]],                         \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                   \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint3 lid [[thread_position_in_threadgroup]]                             \
  );

instantiate_attention_gemm(f16, half, 64, 32) instantiate_attention_gemm(
    f16,
    half,
    128,
    16
) instantiate_attention_gemm(f16, half, 256, 16)

    instantiate_attention_gemm(bf16, bfloat, 64, 32) instantiate_attention_gemm(
        bf16,
        bfloat,
        128,
        16
    ) instantiate_attention_gemm(bf16, bfloat, 256, 16)

        instantiate_attention_gemm(f32, float, 64, 32) instantiate_attention_gemm(
            f32,
            float,
            128,
            16
        ) instantiate_attention_gemm(f32, float, 256, 16)

#undef instantiate_attention_gemm
