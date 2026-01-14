#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#include "../../matmul/common/steel/gemm/gemm.h"
#include "../../matmul/common/steel/gemm/params.h"
#include "../../common/mlp_epilogue.h"

using namespace metal;

namespace steel {

///////////////////////////////////////////////////////////////////////////////
// MLP Fused GEMM Kernel
// Computes paired up/gate projections with fused activation for prefill path.
// Weight layout: B = [up_weights (hidden_dim cols), gate_weights (hidden_dim cols)]
// Output = up * activation(gate), written to first hidden_dim columns
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
    bool K_aligned,
    typename AccumType = typename AccumHelper<T>::accum_type>
struct GEMMMlpFusedKernel {
  STEEL_CONST short tgp_padding_a = 16 / sizeof(T);
  STEEL_CONST short tgp_padding_b = 16 / sizeof(T);
  STEEL_CONST short tgp_mem_size_a =
      transpose_a ? BK * (BM + tgp_padding_a) : BM * (BK + tgp_padding_a);
  // Double B threadgroup memory for up and gate
  STEEL_CONST short tgp_mem_size_b =
      2 * (transpose_b ? BN * (BK + tgp_padding_b) : BK * (BN + tgp_padding_b));
  STEEL_CONST short tgp_mem_size = tgp_mem_size_a + tgp_mem_size_b;

  STEEL_CONST short tgp_size = WM * WN * 32;

  using loader_a_t = BlockLoader<
      T,
      transpose_a ? BK : BM,
      transpose_a ? BM : BK,
      transpose_a ? BM + tgp_padding_a : BK + tgp_padding_a,
      !transpose_a,
      tgp_size>;
  using loader_b_t = BlockLoader<
      T,
      transpose_b ? BN : BK,
      transpose_b ? BK : BN,
      transpose_b ? BK + tgp_padding_b : BN + tgp_padding_b,
      transpose_b,
      tgp_size>;
  using mma_t = BlockMMA<
      T,
      U,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      transpose_a ? BM + tgp_padding_a : BK + tgp_padding_a,
      transpose_b ? BK + tgp_padding_b : BN + tgp_padding_b,
      AccumType,
      TransformNone<U, AccumType>>;

  static METAL_FUNC void run(
      const device T* A [[buffer(0)]],
      const device T* B [[buffer(1)]],
      device U* D [[buffer(2)]],
      const constant GEMMParams* params [[buffer(3)]],
      const constant int& hidden_dim [[buffer(10)]],
      threadgroup T* As [[threadgroup(0)]],
      threadgroup T* Bs_up [[threadgroup(1)]],
      uint simd_lane_id [[thread_index_in_simdgroup]],
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]]) {
    (void)lid;

    const int tid_y = ((tid.y) << params->swizzle_log) +
        ((tid.x) & ((1 << params->swizzle_log) - 1));
    const int tid_x = (tid.x) >> params->swizzle_log;

    // Only process tiles within hidden_dim columns (output size)
    int hidden_tiles = (hidden_dim + BN - 1) / BN;
    if (hidden_tiles <= tid_x || params->tiles_m <= tid_y) {
      return;
    }

    threadgroup_barrier(mem_flags::mem_none);

    const int c_row = tid_y * BM;
    const int c_col = tid_x * BN;
    const size_t c_row_long = size_t(c_row);
    const size_t c_col_long = size_t(c_col);

    // Pointer to A (input activations)
    const device T* A_ptr = A + (transpose_a ? c_row_long : c_row_long * params->lda);

    // Pointers to up and gate weights in B
    // B layout: [up_weights | gate_weights] where each is hidden_dim columns
    const device T* B_up = B + (transpose_b ? c_col_long * params->ldb : c_col_long);
    const device T* B_gate = B_up + (transpose_b ? hidden_dim * params->ldb : hidden_dim);

    device U* D_ptr = D + c_row_long * params->ldd + c_col_long;

    // Threadgroup memory layout: [As, Bs_up, Bs_gate]
    threadgroup T* Bs_gate = Bs_up + (tgp_mem_size_b / 2);

    // Loaders for A (shared) and B (up and gate)
    thread loader_a_t loader_a(A_ptr, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b_up(B_up, params->ldb, Bs_up, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b_gate(B_gate, params->ldb, Bs_gate, simd_group_id, simd_lane_id);

    // MMA operations for up and gate
    thread mma_t mma_up(simd_group_id, simd_lane_id);
    thread mma_t mma_gate(simd_group_id, simd_lane_id);

    int gemm_k_iterations = params->gemm_k_iterations_aligned;

    // Main compute loop
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Load A (shared for both up and gate)
      if (MN_aligned) {
        loader_a.load_unsafe();
      } else {
        short tgp_bm = min(short(BM), short(params->M - c_row));
        short2 tile_dims_A = transpose_a ? short2(tgp_bm, BK) : short2(BK, tgp_bm);
        loader_a.load_safe(tile_dims_A);
      }

      // Load B_up and B_gate
      if (MN_aligned) {
        loader_b_up.load_unsafe();
        loader_b_gate.load_unsafe();
      } else {
        short tgp_bn = min(short(BN), short(hidden_dim - c_col));
        short2 tile_dims_B = transpose_b ? short2(BK, tgp_bn) : short2(tgp_bn, BK);
        loader_b_up.load_safe(tile_dims_B);
        loader_b_gate.load_safe(tile_dims_B);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // MMA for up and gate
      mma_up.mma(As, Bs_up);
      mma_gate.mma(As, Bs_gate);

      loader_a.next();
      loader_b_up.next();
      loader_b_gate.next();
    }

    // Handle leftover K
    if (!K_aligned) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      int lbk = params->K - params->gemm_k_iterations_aligned * BK;
      short2 tile_dims_A = transpose_a ? short2(BM, lbk) : short2(lbk, BM);
      short2 tile_dims_B = transpose_b ? short2(lbk, BN) : short2(BN, lbk);

      loader_a.load_safe(tile_dims_A);
      loader_b_up.load_safe(tile_dims_B);
      loader_b_gate.load_safe(tile_dims_B);

      threadgroup_barrier(mem_flags::mem_threadgroup);

      mma_up.mma(As, Bs_up);
      mma_gate.mma(As, Bs_gate);
    }

    // Apply MLP fused epilogue: out = up * activation(gate)
    // Access the Ctile elements and apply fusion
    constexpr int kElemsPerTile = decltype(mma_up.Ctile)::kElemsPerTile;

    #pragma unroll
    for (short i = 0; i < kElemsPerTile; i++) {
      float up_val = static_cast<float>(mma_up.Ctile.elems()[i]);
      float gate_val = static_cast<float>(mma_gate.Ctile.elems()[i]);
      float fused = mlp_fused_epilogue_f32(up_val, gate_val);
      mma_up.Ctile.elems()[i] = static_cast<AccumType>(fused);
    }

    // Store result
    if (MN_aligned) {
      mma_up.store_result(D_ptr, params->ldd);
    } else {
      short tgp_bm = min(short(BM), short(params->M - c_row));
      short tgp_bn = min(short(BN), short(hidden_dim - c_col));
      mma_up.store_result_safe(D_ptr, params->ldd, short2(tgp_bn, tgp_bm));
    }
  }
};

} // namespace steel
