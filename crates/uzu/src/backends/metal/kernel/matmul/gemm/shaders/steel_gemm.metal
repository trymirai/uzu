// clang-format off
#include "../../../common/utils.h"
#include "../../../definitions.metal"

#include "../../common/steel/gemm/gemm.h"

using namespace steel;

// Upper bounds for threadgroup memory (in elements of T).
// Max across all tile/type combos: 64x32x32 with half (padding=8).
// tgp_mem_size_a = BM * (BK + padding) = 64 * (32 + 8) = 2560
// tgp_mem_size_b = BN * (BK + padding) = 32 * (32 + 8) = 1280
// For 64x64x16 with half: a=64*(16+8)=1536, b=64*(16+8)=1536
// Max a = 2560, max b = 1536
#define GEMM_MAX_TGP_A 2560
#define GEMM_MAX_TGP_B 1536

namespace uzu {
namespace matmul {
using GEMMParams = steel::GEMMParams;
} // namespace matmul
} // namespace uzu

///////////////////////////////////////////////////////////////////////////////
// GEMM implementation
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename AccumType = float>
METAL_FUNC void gemm_impl(
    const device T* a,
    const device T* b,
    device T* d,
    const constant GEMMParams* params,
    const int BM,
    const int BN,
    const int BK,
    const int WM,
    const int WN,
    const bool align_m,
    const bool align_n,
    const bool align_k,
    threadgroup T* a_shared,
    threadgroup T* b_shared,
    uint simd_lane_id,
    uint simd_group_id,
    uint3 tid,
    uint3 lid
) {
  (void)lid;

  // Hardcoded: transpose_a = false, transpose_b = true
  const bool transpose_a = false;
  const bool transpose_b = true;

  const short tgp_padding = 16 / sizeof(T);
  const short lda_tgp = BK + tgp_padding;  // transpose_a=false: BK + pad
  const short ldb_tgp = BK + tgp_padding;  // transpose_b=true:  BK + pad
  const short tgp_size = WM * WN * 32;

  // Find block
  const int tid_y = ((tid.y) << params->swizzle_log) +
                    ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  // Batch offset (non-batched path only)
  a += params->batch_stride_a * tid.z;
  b += params->batch_stride_b * tid.z;
  d += params->batch_stride_d * tid.z;

  threadgroup_barrier(mem_flags::mem_none);

  // Find block in a, b, d
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  // transpose_a=false: a += c_row * lda
  a += c_row_long * params->lda;
  // transpose_b=true: b += c_col * ldb
  b += c_col_long * params->ldb;
  d += c_row_long * params->ldd + c_col_long;

  // Construct loader and MMA objects with runtime tile params
  // transpose_a=false: loader_a BROWS=BM, BCOLS=BK, dst_ld=BK+pad, reduction_dim=1
  thread BlockLoader<T> loader_a(
      a, params->lda, a_shared,
      simd_group_id, simd_lane_id,
      BM, BK, lda_tgp, 1, tgp_size);

  // transpose_b=true: loader_b BROWS=BN, BCOLS=BK, dst_ld=BK+pad, reduction_dim=1
  thread BlockLoader<T> loader_b(
      b, params->ldb, b_shared,
      simd_group_id, simd_lane_id,
      BN, BK, ldb_tgp, 1, tgp_size);

  thread BlockMMA<T, T, AccumType> mma_op(
      simd_group_id, simd_lane_id,
      BM, BN, BK, WM, WN,
      transpose_a, transpose_b,
      lda_tgp, ldb_tgp);

  const short tgp_bm = align_m ? BM : short(min(BM, params->M - c_row));
  const short tgp_bn = align_n ? BN : short(min(BN, params->N - c_col));

  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  // Do unaligned K iterations first
  if (!align_k) {
    const int k_last = params->gemm_k_iterations_aligned * BK;
    const int k_remain = params->K - k_last;
    // transpose_a=false: k_jump_a = k_last
    const size_t k_jump_a = size_t(k_last);
    // transpose_b=true: k_jump_b = k_last
    const size_t k_jump_b = size_t(k_last);

    loader_a.src += k_jump_a;
    loader_b.src += k_jump_b;

    // transpose_a=false: tile_dims_a = (k_remain, tgp_bm)
    const short2 tile_dims_a = short2(k_remain, tgp_bm);
    // transpose_b=true: tile_dims_b = (k_remain, tgp_bn)
    const short2 tile_dims_b = short2(k_remain, tgp_bn);

    loader_a.load_safe(tile_dims_a);
    loader_b.load_safe(tile_dims_b);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(a_shared, b_shared);

    loader_a.src -= k_jump_a;
    loader_b.src -= k_jump_b;
  }

  // MNK aligned loop
  if (align_m && align_n) {
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(a_shared, b_shared);
      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);
    return mma_op.store_result(d, params->ldd);
  } else {
    const short leftover_bk = 0;

    if ((align_m || tgp_bm == BM) && (align_n || tgp_bn == BN)) {
      gemm_loop<T, T, AccumType>(
          a_shared, b_shared, gemm_k_iterations, loader_a, loader_b,
          mma_op, tgp_bm, tgp_bn, leftover_bk,
          BK, transpose_a, transpose_b, true, true, true);
      return mma_op.store_result(d, params->ldd);
    } else if (align_n || tgp_bn == BN) {
      gemm_loop<T, T, AccumType>(
          a_shared, b_shared, gemm_k_iterations, loader_a, loader_b,
          mma_op, tgp_bm, tgp_bn, leftover_bk,
          BK, transpose_a, transpose_b, false, true, true);
      return mma_op.store_result_safe(d, params->ldd, short2(tgp_bn, tgp_bm));
    } else if (align_m || tgp_bm == BM) {
      gemm_loop<T, T, AccumType>(
          a_shared, b_shared, gemm_k_iterations, loader_a, loader_b,
          mma_op, tgp_bm, tgp_bn, leftover_bk,
          BK, transpose_a, transpose_b, true, false, true);
      return mma_op.store_result_safe(d, params->ldd, short2(tgp_bn, tgp_bm));
    } else {
      gemm_loop<T, T, AccumType>(
          a_shared, b_shared, gemm_k_iterations, loader_a, loader_b,
          mma_op, tgp_bm, tgp_bn, leftover_bk,
          BK, transpose_a, transpose_b, false, false, true);
      return mma_op.store_result_safe(d, params->ldd, short2(tgp_bn, tgp_bm));
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Unified DSL kernel
///////////////////////////////////////////////////////////////////////////////

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemm)(
    const device T* a,
    const device T* b,
    device T* d,
    const constant uzu::matmul::GEMMParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    threadgroup T a_shared[GEMM_MAX_TGP_A],
    threadgroup T b_shared[GEMM_MAX_TGP_B],
    const uint block_rows SPECIALIZE,
    const uint block_cols SPECIALIZE,
    const uint block_depth SPECIALIZE,
    const uint warps_per_row SPECIALIZE,
    const uint warps_per_col SPECIALIZE,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool align_k SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(2),
    const uint thread_z THREADS(2),
    const Simd simd
) {
  if (simd.group_idx >= warps_per_row * warps_per_col) {
    return;
  }

  gemm_impl<T, float>(
      a, b, d, params,
      block_rows, block_cols, block_depth,
      warps_per_row, warps_per_col,
      align_m, align_n, align_k,
      a_shared, b_shared,
      simd.lane_idx, simd.group_idx,
      uint3(group_x, group_y, group_z),
      uint3(thread_x, thread_y, thread_z)
  );
}

// clang-format on
