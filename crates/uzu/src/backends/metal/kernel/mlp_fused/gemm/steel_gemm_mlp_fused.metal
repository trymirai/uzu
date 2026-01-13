#include <metal_stdlib>

#include "../../common/utils.h"
#include "steel_gemm_mlp_fused.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// MLP Fused GEMM Kernel Instantiations
// For prefill path: computes up * activation(gate) in a single pass
///////////////////////////////////////////////////////////////////////////////

#define instantiate_mlp_fused_gemm(                                           \
    tname, trans_a, trans_b, itype, otype, bm, bn, bk, wm, wn, aname, mn, k)  \
  template [[host_name("steel_gemm_mlp_fused_" #tname "_" #itype "_" #otype   \
                       "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn      \
                       "_" #aname "_MN_" #mn "_K_" #k)]] [[kernel]] void      \
  steel_gemm_mlp_fused<                                                       \
      itype, otype, bm, bn, bk, wm, wn, trans_a, trans_b, mn, k>(             \
      const device itype* A [[buffer(0)]],                                    \
      const device itype* B [[buffer(1)]],                                    \
      device otype* D [[buffer(2)]],                                          \
      const constant steel::GEMMParams* params [[buffer(3)]],                 \
      const constant int& hidden_dim [[buffer(10)]],                          \
      uint simd_lane_id [[thread_index_in_simdgroup]],                        \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                  \
      uint3 tid [[threadgroup_position_in_grid]],                             \
      uint3 lid [[thread_position_in_threadgroup]])

template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    bool MN_aligned,
    bool K_aligned>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
steel_gemm_mlp_fused(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device U* D [[buffer(2)]],
    const constant steel::GEMMParams* params [[buffer(3)]],
    const constant int& hidden_dim [[buffer(10)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using kernel_t =
      steel::GEMMMlpFusedKernel<T, U, BM, BN, BK, WM, WN, transpose_a, transpose_b, MN_aligned, K_aligned>;

  // Allocate threadgroup memory
  threadgroup T As[kernel_t::tgp_mem_size_a];
  threadgroup T Bs[kernel_t::tgp_mem_size_b];

  kernel_t::run(
      A, B, D, params, hidden_dim, As, Bs, simd_lane_id, simd_group_id, tid, lid);
}

// Instantiate for common configurations
// transpose_a = false, transpose_b = true (weights are [K, 2*H], accessed as transposed)

// float16 configurations
instantiate_mlp_fused_gemm(t, false, true, half, half, 32, 32, 16, 2, 2, align, true, true);
instantiate_mlp_fused_gemm(t, false, true, half, half, 32, 32, 16, 2, 2, align, true, false);
instantiate_mlp_fused_gemm(t, false, true, half, half, 32, 32, 16, 2, 2, align, false, true);
instantiate_mlp_fused_gemm(t, false, true, half, half, 32, 32, 16, 2, 2, align, false, false);

instantiate_mlp_fused_gemm(t, false, true, half, half, 64, 64, 16, 2, 2, align, true, true);
instantiate_mlp_fused_gemm(t, false, true, half, half, 64, 64, 16, 2, 2, align, true, false);
instantiate_mlp_fused_gemm(t, false, true, half, half, 64, 64, 16, 2, 2, align, false, true);
instantiate_mlp_fused_gemm(t, false, true, half, half, 64, 64, 16, 2, 2, align, false, false);

// bfloat16 configurations
instantiate_mlp_fused_gemm(t, false, true, bfloat, bfloat, 32, 32, 16, 2, 2, align, true, true);
instantiate_mlp_fused_gemm(t, false, true, bfloat, bfloat, 32, 32, 16, 2, 2, align, true, false);
instantiate_mlp_fused_gemm(t, false, true, bfloat, bfloat, 32, 32, 16, 2, 2, align, false, true);
instantiate_mlp_fused_gemm(t, false, true, bfloat, bfloat, 32, 32, 16, 2, 2, align, false, false);

instantiate_mlp_fused_gemm(t, false, true, bfloat, bfloat, 64, 64, 16, 2, 2, align, true, true);
instantiate_mlp_fused_gemm(t, false, true, bfloat, bfloat, 64, 64, 16, 2, 2, align, true, false);
instantiate_mlp_fused_gemm(t, false, true, bfloat, bfloat, 64, 64, 16, 2, 2, align, false, true);
instantiate_mlp_fused_gemm(t, false, true, bfloat, bfloat, 64, 64, 16, 2, 2, align, false, false);

// float32 configurations
instantiate_mlp_fused_gemm(t, false, true, float, float, 32, 32, 16, 2, 2, align, true, true);
instantiate_mlp_fused_gemm(t, false, true, float, float, 32, 32, 16, 2, 2, align, true, false);
instantiate_mlp_fused_gemm(t, false, true, float, float, 32, 32, 16, 2, 2, align, false, true);
instantiate_mlp_fused_gemm(t, false, true, float, float, 32, 32, 16, 2, 2, align, false, false);

// Non-transposed B (weights are [2*H, K])
instantiate_mlp_fused_gemm(n, false, false, half, half, 32, 32, 16, 2, 2, align, true, true);
instantiate_mlp_fused_gemm(n, false, false, half, half, 32, 32, 16, 2, 2, align, true, false);
instantiate_mlp_fused_gemm(n, false, false, half, half, 32, 32, 16, 2, 2, align, false, true);
instantiate_mlp_fused_gemm(n, false, false, half, half, 32, 32, 16, 2, 2, align, false, false);

instantiate_mlp_fused_gemm(n, false, false, bfloat, bfloat, 32, 32, 16, 2, 2, align, true, true);
instantiate_mlp_fused_gemm(n, false, false, bfloat, bfloat, 32, 32, 16, 2, 2, align, true, false);
instantiate_mlp_fused_gemm(n, false, false, bfloat, bfloat, 32, 32, 16, 2, 2, align, false, true);
instantiate_mlp_fused_gemm(n, false, false, bfloat, bfloat, 32, 32, 16, 2, 2, align, false, false);

instantiate_mlp_fused_gemm(n, false, false, float, float, 32, 32, 16, 2, 2, align, true, true);
instantiate_mlp_fused_gemm(n, false, false, float, float, 32, 32, 16, 2, 2, align, true, false);
instantiate_mlp_fused_gemm(n, false, false, float, float, 32, 32, 16, 2, 2, align, false, true);
instantiate_mlp_fused_gemm(n, false, false, float, float, 32, 32, 16, 2, 2, align, false, false);
