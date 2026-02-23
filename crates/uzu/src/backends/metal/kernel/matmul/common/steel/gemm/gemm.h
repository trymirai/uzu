#pragma once

#include "loader.h"
#include "mma.h"
#include "params.h"
#include "transforms.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernel helpers
///////////////////////////////////////////////////////////////////////////////

namespace steel {

template <
    typename T,
    typename U,
    typename AccumType = float,
    typename Epilogue = TransformNone<U, AccumType>>
METAL_FUNC void gemm_loop(
    threadgroup T* As,
    threadgroup T* Bs,
    const int gemm_k_iterations,
    thread BlockLoader<T>& loader_a,
    thread BlockLoader<T>& loader_b,
    thread BlockMMA<T, U, AccumType, Epilogue>& mma_op,
    thread const short& tgp_bm,
    thread const short& tgp_bn,
    thread const short& lbk,
    const short BK,
    const bool transpose_a,
    const bool transpose_b,
    const bool align_m,
    const bool align_n,
    const bool align_k
) {
  short2 tile_dims_A = transpose_a ? short2(tgp_bm, BK) : short2(BK, tgp_bm);
  short2 tile_dims_B = transpose_b ? short2(BK, tgp_bn) : short2(tgp_bn, BK);

  for (int k = 0; k < gemm_k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (align_m) {
      loader_a.load_unsafe();
    } else {
      loader_a.load_safe(tile_dims_A);
    }

    if (align_n) {
      loader_b.load_unsafe();
    } else {
      loader_b.load_safe(tile_dims_B);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.next();
    loader_b.next();
  }

  if (!align_k) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    short2 tile_dims_A_last =
        transpose_a ? short2(tgp_bm, lbk) : short2(lbk, tgp_bm);
    short2 tile_dims_B_last =
        transpose_b ? short2(lbk, tgp_bn) : short2(tgp_bn, lbk);

    loader_a.load_safe(tile_dims_A_last);
    loader_b.load_safe(tile_dims_B_last);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
  }
}

} // namespace steel
