#include <metal_simdgroup>
#include <metal_stdlib>

#include "../../common/utils.h"
#include "../../common/mlp_epilogue.h"

#include "../matmul/common/steel/utils.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
/// MLP Fused GEMV - Paired Up/Gate Computation with Activation Fusion
///////////////////////////////////////////////////////////////////////////////

#define MTL_CONST static constant constexpr const

// MLP fused GEMV kernel for decode path (M=1).
// Computes paired up and gate projections, then applies: out = up * activation(gate)
// Output size is hidden_dim (half of the weight matrix columns).
template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
struct GEMVMlpFusedKernel {
  using AccT = float;

  MTL_CONST int threadsM = BM * SM;
  MTL_CONST int threadsN = BN * SN;
  MTL_CONST int blockM = threadsM * TM;
  MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup can only have 32 threads");

  MTL_CONST short tgp_mem_size = BN > 1 ? 2 * BN * (blockM + TM) : 0;
  MTL_CONST bool needs_tgp_reduction = BN > 1;

  template <typename U = T>
  static METAL_FUNC void
  load_unsafe(const device T* src, thread U dst[TN], const int src_offset = 0) {
    MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      dst[tn] = static_cast<U>(src[src_offset + tn]);
    }
  }

  template <typename U = T>
  static METAL_FUNC void load_safe(
      const device T* src,
      thread U dst[TN],
      const int src_offset = 0,
      const int src_size = TN) {
    if (src_offset + TN <= src_size) {
      MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = static_cast<U>(src[src_offset + tn]);
      }
    } else {
      MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = src_offset + tn < src_size
            ? static_cast<U>(src[src_offset + tn])
            : U(0);
      }
    }
  }

  static METAL_FUNC void run(
      const device T* mat [[buffer(0)]],
      const device T* in_vec [[buffer(1)]],
      device T* out_vec [[buffer(3)]],
      const int in_vec_size,
      const int hidden_dim,  // Output size (half of weight matrix rows)
      const int matrix_ld,
      threadgroup AccT* tgp_memory,
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    (void)lid;

    // Thread local accumulation results for up and gate
    thread AccT result_up[TM] = {0};
    thread AccT result_gate[TM] = {0};
    thread T inter_up[TN];
    thread T inter_gate[TN];
    thread AccT v_coeff[TN];

    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);

    const int sgN = BN != 1 ? (simd_gid % BN) : 0;
    const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
    const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;

    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;

    // Block position - outputs to hidden_dim elements
    int out_row = tid.x * blockM + bm;

    if (out_row >= hidden_dim)
      return;

    out_row = out_row + TM <= hidden_dim ? out_row : hidden_dim - TM;

    // Pointers to up and gate rows in the weight matrix
    // Weight layout: [up_0, up_1, ..., up_H-1, gate_0, gate_1, ..., gate_H-1]
    const device T* mat_up = mat + out_row * matrix_ld;
    const device T* mat_gate = mat + (out_row + hidden_dim) * matrix_ld;

    constexpr const uniform<int> loop_stride = make_uniform(blockN);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    const uniform<int> last_iter = loop_stride * n_iter;
    const uniform<int> leftover = in_size - last_iter;

    // Main loop - compute both up and gate dot products
    for (int i = 0; i < n_iter; ++i) {
      load_unsafe<AccT>(in_vec, v_coeff, bn);

      int mat_offset = 0;
      MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        load_unsafe(mat_up, inter_up, mat_offset + bn);
        load_unsafe(mat_gate, inter_gate, mat_offset + bn);

        MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result_up[tm] += inter_up[tn] * v_coeff[tn];
          result_gate[tm] += inter_gate[tn] * v_coeff[tn];
        }

        mat_offset += matrix_ld;
      }

      bn += blockN;
    }

    // Handle leftover
    if (leftover > 0) {
      load_safe<AccT>(in_vec, v_coeff, bn, in_vec_size);

      MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        load_safe(&mat_up[tm * matrix_ld], inter_up, bn, in_vec_size);
        load_safe(&mat_gate[tm * matrix_ld], inter_gate, bn, in_vec_size);

        MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result_up[tm] += inter_up[tn] * v_coeff[tn];
          result_gate[tm] += inter_gate[tn] * v_coeff[tn];
        }
      }
    }

    // Simdgroup reduction for both up and gate
    MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
      MTL_PRAGMA_UNROLL
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
        result_up[tm] += simd_shuffle_down(result_up[tm], sn);
        result_gate[tm] += simd_shuffle_down(result_gate[tm], sn);
      }
    }

    // Threadgroup reduction if needed
    if (needs_tgp_reduction) {
      threadgroup AccT* tgp_results_up = tgp_memory + sgN * (blockM + TM) + bm;
      threadgroup AccT* tgp_results_gate = tgp_memory + BN * (blockM + TM) + sgN * (blockM + TM) + bm;

      if (thrN == 0) {
        MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          tgp_results_up[tm] = result_up[tm];
          tgp_results_gate[tm] = result_gate[tm];
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (sgN == 0) {
          MTL_PRAGMA_UNROLL
          for (int sgn = 1; sgn < BN; sgn++) {
            MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
              result_up[tm] += tgp_results_up[sgn * (blockM + TM) + tm];
              result_gate[tm] += tgp_results_gate[sgn * (blockM + TM) + tm];
            }
          }
        }
      }
    }

    // Apply MLP fused epilogue and write output
    if (simdN == 0 && thrN == 0) {
      MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        float fused_result = mlp_fused_epilogue_f32(result_up[tm], result_gate[tm]);
        out_vec[out_row + tm] = static_cast<T>(fused_result);
      }
    }
  }
};

template <
    typename T,
    const int BM,
    const int BN,
    const int SM,
    const int SN,
    const int TM,
    const int TN>
[[kernel, max_total_threads_per_threadgroup(BM* BN * 32)]] void gemv_mlp_fused(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& hidden_dim [[buffer(5)]],  // Output dimension (half of weight rows)
    const constant int& matrix_ld [[buffer(6)]],
    const constant int64_t& vector_batch_stride [[buffer(11)]],
    const constant int64_t& matrix_batch_stride [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel = GEMVMlpFusedKernel<T, BM, BN, SM, SN, TM, TN>;
  threadgroup typename gemv_kernel::AccT tgp_memory
      [gemv_kernel::tgp_mem_size == 0 ? 1 : gemv_kernel::tgp_mem_size];

  // Batch offset
  in_vec += tid.z * vector_batch_stride;
  mat += tid.z * matrix_batch_stride;
  out_vec += tid.z * hidden_dim;

  gemv_kernel::run(
      mat,
      in_vec,
      out_vec,
      in_vec_size,
      hidden_dim,
      matrix_ld,
      gemv_kernel::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

#define instantiate_gemv_mlp_fused_helper(name, itype, bm, bn, sm, sn, tm, tn) \
  instantiate_kernel(                                                          \
      "gemv_mlp_fused_" #name "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn "_tm" #tm \
      "_tn" #tn,                                                               \
      gemv_mlp_fused,                                                          \
      itype,                                                                   \
      bm,                                                                      \
      bn,                                                                      \
      sm,                                                                      \
      sn,                                                                      \
      tm,                                                                      \
      tn)

// clang-format off
#define instantiate_gemv_mlp_fused_blocks(name, itype) \
  instantiate_gemv_mlp_fused_helper(name, itype, 4, 1, 1, 32, 4, 4) \
  instantiate_gemv_mlp_fused_helper(name, itype, 8, 1, 1, 32, 4, 4) // clang-format on

instantiate_gemv_mlp_fused_blocks(float32, float);
instantiate_gemv_mlp_fused_blocks(float16, half);
instantiate_gemv_mlp_fused_blocks(bfloat16, bfloat16_t);
