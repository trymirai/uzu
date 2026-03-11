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
    typename AccumulatorType = float,
    typename Epilogue = TransformNone<U, AccumulatorType>>
METAL_FUNC void gemm_loop(
    threadgroup T* left_shared,
    threadgroup T* right_shared,
    const int gemm_k_iterations,
    thread BlockLoader<T>& loader_a,
    thread BlockLoader<T>& loader_b,
    thread BlockMMA<T, U, AccumulatorType, Epilogue>& mma_operation,
    thread const short& threadgroup_block_m,
    thread const short& threadgroup_block_n,
    thread const short& leftover_block_k,
    const short BLOCK_K,
    const bool transpose_a,
    const bool transpose_b,
    const bool align_m,
    const bool align_n,
    const bool align_k
) {
  short2 tile_dimensions_a = transpose_a ? short2(threadgroup_block_m, BLOCK_K) : short2(BLOCK_K, threadgroup_block_m);
  short2 tile_dimensions_b = transpose_b ? short2(BLOCK_K, threadgroup_block_n) : short2(threadgroup_block_n, BLOCK_K);

  for (int k = 0; k < gemm_k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (align_m) {
      loader_a.load_unchecked();
    } else {
      loader_a.load_checked(tile_dimensions_a);
    }

    if (align_n) {
      loader_b.load_unchecked();
    } else {
      loader_b.load_checked(tile_dimensions_b);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_operation.mma(left_shared, right_shared);
    loader_a.next();
    loader_b.next();
  }

  if (!align_k) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    short2 tile_dimensions_a_last =
        transpose_a ? short2(threadgroup_block_m, leftover_block_k) : short2(leftover_block_k, threadgroup_block_m);
    short2 tile_dimensions_b_last =
        transpose_b ? short2(leftover_block_k, threadgroup_block_n) : short2(threadgroup_block_n, leftover_block_k);

    loader_a.load_checked(tile_dimensions_a_last);
    loader_b.load_checked(tile_dimensions_b_last);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_operation.mma(left_shared, right_shared);
  }
}

} // namespace steel
