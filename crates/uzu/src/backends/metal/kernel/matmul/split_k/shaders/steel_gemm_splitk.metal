
#include "../../../common/utils.h"
#include "../../../definitions.metal"
#include "../../common/steel/gemm/gemm.h"

using namespace steel;

// Upper bounds for split-K threadgroup memory.
// Split-K tile: BM=16, BN=32, BK=16 with bfloat16 (padding=8)
// tgp_mem_size_a = BM * (BK + padding) = 16 * (16 + 8) = 384
// tgp_mem_size_b = BN * (BK + padding) = 32 * (16 + 8) = 768
#define SPLITK_MAX_TGP_A 384
#define SPLITK_MAX_TGP_B 768

namespace uzu {
namespace matmul {
using GEMMSpiltKParams = steel::GEMMSpiltKParams;
} // namespace matmul
} // namespace uzu

///////////////////////////////////////////////////////////////////////////////
// Split-K GEMM implementation
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U>
METAL_FUNC void gemm_splitk_impl(
    const device T* a,
    const device T* b,
    device U* c,
    const constant GEMMSpiltKParams* params,
    const int BM,
    const int BN,
    const int BK,
    const int WM,
    const int WN,
    const bool align_m,
    const bool align_k,
    threadgroup T* a_shared,
    threadgroup T* b_shared,
    uint simd_lane_id,
    uint simd_group_id,
    uint3 tid,
    uint3 lid
) {
  (void)lid;

  const bool transpose_a = false;
  const bool transpose_b = true;

  const short tgp_padding = 16 / sizeof(T);
  const short lda_tgp = BK + tgp_padding;
  const short ldb_tgp = BK + tgp_padding;
  const short tgp_size = WM * WN * 32;

  const int tid_x = tid.x;
  const int tid_y = tid.y;
  const int tid_z = tid.z;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int k_start = params->split_k_partition_size * tid_z;

  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);
  const size_t k_start_long = size_t(k_start);

  a += k_start_long + c_row_long * params->lda;
  b += k_start_long + c_col_long * params->ldb;
  c += (size_t(params->split_k_partition_stride) * tid_z) +
       (c_row_long * params->ldc + c_col_long);

  thread BlockLoader<T> loader_a(
      a, params->lda, a_shared,
      simd_group_id, simd_lane_id,
      BM, BK, lda_tgp, 1, tgp_size);

  thread BlockLoader<T> loader_b(
      b, params->ldb, b_shared,
      simd_group_id, simd_lane_id,
      BN, BK, ldb_tgp, 1, tgp_size);

  thread BlockMMA<T, U, float> mma_op(
      simd_group_id, simd_lane_id,
      BM, BN, BK, WM, WN,
      transpose_a, transpose_b,
      lda_tgp, ldb_tgp);

  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  short tgp_bm = min(BM, params->M - c_row);
  short tgp_bn = min(BN, params->N - c_col);
  short leftover_bk = params->K % BK;

  bool mn_aligned = align_m || (tgp_bm == BM && tgp_bn == BN);

  if (mn_aligned) {
    gemm_loop<T, U, float>(
        a_shared, b_shared, gemm_k_iterations, loader_a, loader_b,
        mma_op, tgp_bm, tgp_bn, leftover_bk,
        BK, transpose_a, transpose_b, true, true, true);
  } else if (tgp_bn == BN) {
    gemm_loop<T, U, float>(
        a_shared, b_shared, gemm_k_iterations, loader_a, loader_b,
        mma_op, tgp_bm, tgp_bn, leftover_bk,
        BK, transpose_a, transpose_b, false, true, true);
  } else if (tgp_bm == BM) {
    gemm_loop<T, U, float>(
        a_shared, b_shared, gemm_k_iterations, loader_a, loader_b,
        mma_op, tgp_bm, tgp_bn, leftover_bk,
        BK, transpose_a, transpose_b, true, false, true);
  } else {
    gemm_loop<T, U, float>(
        a_shared, b_shared, gemm_k_iterations, loader_a, loader_b,
        mma_op, tgp_bm, tgp_bn, leftover_bk,
        BK, transpose_a, transpose_b, false, false, true);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if ((tid_z + 1) == (params->split_k_partitions)) {
    int gemm_k_iter_remaining =
        (params->K - (k_start + params->split_k_partition_size)) / BK;
    if (!align_k || gemm_k_iter_remaining > 0)
      gemm_loop<T, U, float>(
          a_shared, b_shared, gemm_k_iter_remaining, loader_a, loader_b,
          mma_op, tgp_bm, tgp_bn, leftover_bk,
          BK, transpose_a, transpose_b, false, false, align_k);
  }

  if (mn_aligned) {
    mma_op.store_result(c, params->ldc);
  } else {
    mma_op.store_result_safe(c, params->ldc, short2(tgp_bn, tgp_bm));
  }
}

///////////////////////////////////////////////////////////////////////////////
// Split-K accumulation
///////////////////////////////////////////////////////////////////////////////

template <
    typename AccT,
    typename OutT,
    typename Epilogue = TransformNone<OutT, AccT>>
METAL_FUNC void gemm_splitk_accum_impl(
    const device AccT* c_split,
    device OutT* d,
    const constant int& k_partitions,
    const constant int& partition_stride,
    const constant int& ldd,
    uint2 gid
) {
  d += gid.x + gid.y * size_t(ldd);
  c_split += gid.x + gid.y * size_t(ldd);

  size_t offset = 0;
  AccT out = 0;

  for (int i = 0; i < k_partitions; i++) {
    out += c_split[offset];
    offset += partition_stride;
  }

  d[0] = Epilogue::apply(out);
}

///////////////////////////////////////////////////////////////////////////////
// DSL kernel entry points
///////////////////////////////////////////////////////////////////////////////

KERNEL(MatmulSplitKPartialBfloat16)(
    const device bfloat16_t* a,
    const device bfloat16_t* b,
    device float* c,
    const constant uzu::matmul::GEMMSpiltKParams* params,
    const constant uint& partial_group_count_x,
    const constant uint& partial_group_count_y,
    const constant uint& partial_group_count_z,
    const uint group_x GROUPS(partial_group_count_x),
    const uint group_y GROUPS(partial_group_count_y),
    const uint group_z GROUPS(partial_group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(2),
    const uint thread_z THREADS(2),
    threadgroup bfloat16_t a_shared[SPLITK_MAX_TGP_A],
    threadgroup bfloat16_t b_shared[SPLITK_MAX_TGP_B],
    const Simd simd
) {
  gemm_splitk_impl<bfloat16_t, float>(
      a, b, c, params,
      16, 32, 16, 2, 2,
      false, true,
      a_shared, b_shared,
      simd.lane_idx, simd.group_idx,
      uint3(group_x, group_y, group_z),
      uint3(thread_x, thread_y, thread_z)
  );
}

KERNEL(MatmulSplitKAccumBfloat16)(
    const device float* c_split,
    device bfloat16_t* d,
    const constant int& k_partitions,
    const constant int& partition_stride,
    const constant int& ldd,
    const constant uint& accum_total_threads_x,
    const constant uint& accum_total_threads_y,
    const uint gid_x AXIS(accum_total_threads_x, 16),
    const uint gid_y AXIS(accum_total_threads_y, 16)
) {
  gemm_splitk_accum_impl<float, bfloat16_t>(
      c_split, d, k_partitions, partition_stride, ldd, uint2(gid_x, gid_y)
  );
}
