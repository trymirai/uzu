#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#include "../../matmul/common/steel/gemm/gemm.h"
#include "../../matmul/common/shared_types.h"
#include "../../common/mlp_epilogue.h"

using namespace metal;
using namespace steel;

///////////////////////////////////////////////////////////////////////////////
// MLP Fused Split-K GEMM Kernel
// Computes paired up/gate projections with fused activation using split-K.
///////////////////////////////////////////////////////////////////////////////

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
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
gemm_splitk_mlp_fused(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device U* C [[buffer(2)]],
    const constant uzu::matmul::GEMMSpiltKMlpFusedParams* params [[buffer(3)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
  (void)lid;

  using gemm_kernel = GEMMKernel<
      T,
      U,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      MN_aligned,
      K_aligned>;
  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs_up[gemm_kernel::tgp_mem_size_b];
  threadgroup T Bs_gate[gemm_kernel::tgp_mem_size_b];

  const int tid_x = tid.x;
  const int tid_y = tid.y;
  const int tid_z = tid.z;

  // Only process tiles within hidden_dim columns
  int hidden_tiles = (params->hidden_dim + BN - 1) / BN;
  if (hidden_tiles <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  // Find block in A, B, C
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int k_start = params->split_k_partition_size * tid_z;

  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);
  const size_t k_start_long = size_t(k_start);

  // Pointers to A (input)
  const device T* A_ptr =
      A + (transpose_a ? (c_row_long + k_start_long * params->lda)
                       : (k_start_long + c_row_long * params->lda));

  // Pointers to B (up and gate weights)
  // B layout: [up_weights | gate_weights] where each is hidden_dim columns
  const device T* B_up =
      B + (transpose_b ? (k_start_long + c_col_long * params->ldb)
                       : (c_col_long + k_start_long * params->ldb));
  const device T* B_gate =
      B_up +
      (transpose_b ? params->hidden_dim * params->ldb : params->hidden_dim);

  // C output for partial sums (up and gate interleaved in split_k dimension)
  // Layout: [up_partitions, gate_partitions] in the partition stride
  device U* C_up = C + (size_t(params->split_k_partition_stride) * tid_z) +
                   (c_row_long * params->ldc + c_col_long);
  device U* C_gate = C_up + (size_t(params->split_k_partition_stride) *
                             params->split_k_partitions);

  // Loaders
  thread loader_a_t
      loader_a(A_ptr, params->lda, As, simd_group_id, simd_lane_id);
  thread loader_b_t
      loader_b_up(B_up, params->ldb, Bs_up, simd_group_id, simd_lane_id);
  thread loader_b_t
      loader_b_gate(B_gate, params->ldb, Bs_gate, simd_group_id, simd_lane_id);

  // MMA operations
  thread mma_t mma_up(simd_group_id, simd_lane_id);
  thread mma_t mma_gate(simd_group_id, simd_lane_id);

  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  short tgp_bm = min(BM, params->M - c_row);
  short tgp_bn = min(BN, params->hidden_dim - c_col);
  short leftover_bk = params->K % BK;

  // Main loop
  for (int k = 0; k < gemm_k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (MN_aligned) {
      loader_a.load_unsafe();
      loader_b_up.load_unsafe();
      loader_b_gate.load_unsafe();
    } else if (tgp_bm == BM && tgp_bn == BN) {
      loader_a.load_unsafe();
      loader_b_up.load_unsafe();
      loader_b_gate.load_unsafe();
    } else {
      short2 tile_dims_A =
          transpose_a ? short2(tgp_bm, BK) : short2(BK, tgp_bm);
      short2 tile_dims_B =
          transpose_b ? short2(BK, tgp_bn) : short2(tgp_bn, BK);
      loader_a.load_safe(tile_dims_A);
      loader_b_up.load_safe(tile_dims_B);
      loader_b_gate.load_safe(tile_dims_B);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    mma_up.mma(As, Bs_up);
    mma_gate.mma(As, Bs_gate);

    loader_a.next();
    loader_b_up.next();
    loader_b_gate.next();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Handle remaining K for last partition
  if ((tid_z + 1) == (params->split_k_partitions)) {
    int gemm_k_iter_remaining =
        (params->K - (k_start + params->split_k_partition_size)) / BK;
    for (int k = 0; k < gemm_k_iter_remaining; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (!K_aligned) {
        short2 tile_dims_A = transpose_a ? short2(tgp_bm, leftover_bk)
                                         : short2(leftover_bk, tgp_bm);
        short2 tile_dims_B = transpose_b ? short2(leftover_bk, tgp_bn)
                                         : short2(tgp_bn, leftover_bk);
        loader_a.load_safe(tile_dims_A);
        loader_b_up.load_safe(tile_dims_B);
        loader_b_gate.load_safe(tile_dims_B);
      } else {
        loader_a.load_unsafe();
        loader_b_up.load_unsafe();
        loader_b_gate.load_unsafe();
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      mma_up.mma(As, Bs_up);
      mma_gate.mma(As, Bs_gate);

      loader_a.next();
      loader_b_up.next();
      loader_b_gate.next();
    }
  }

  // Store partial results
  if (MN_aligned || (tgp_bm == BM && tgp_bn == BN)) {
    mma_up.store_result(C_up, params->ldc);
    mma_gate.store_result(C_gate, params->ldc);
  } else {
    mma_up.store_result_safe(C_up, params->ldc, short2(tgp_bn, tgp_bm));
    mma_gate.store_result_safe(C_gate, params->ldc, short2(tgp_bn, tgp_bm));
  }
}

///////////////////////////////////////////////////////////////////////////////
// MLP Fused Split-K accumulation kernel
// Accumulates partial sums and applies MLP fused epilogue
///////////////////////////////////////////////////////////////////////////////

template <typename AccT, typename OutT>
[[kernel]] void gemm_splitk_mlp_fused_accum(
    const device AccT* C_split [[buffer(0)]],
    device OutT* D [[buffer(1)]],
    const constant int& k_partitions [[buffer(2)]],
    const constant int& partition_stride [[buffer(3)]],
    const constant int& ldd [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
  // Adjust pointers
  D += gid.x + gid.y * size_t(ldd);
  const device AccT* C_up = C_split + gid.x + gid.y * size_t(ldd);
  const device AccT* C_gate = C_up + (size_t(partition_stride) * k_partitions);

  // Accumulate up and gate partial sums
  AccT out_up = 0;
  AccT out_gate = 0;

  size_t offset = 0;
  for (int i = 0; i < k_partitions; i++) {
    out_up += C_up[offset];
    out_gate += C_gate[offset];
    offset += partition_stride;
  }

  // Apply MLP fused epilogue: out = up * activation(gate)
  float fused = mlp_fused_epilogue_f32(
      static_cast<float>(out_up),
      static_cast<float>(out_gate)
  );
  D[0] = static_cast<OutT>(fused);
}
