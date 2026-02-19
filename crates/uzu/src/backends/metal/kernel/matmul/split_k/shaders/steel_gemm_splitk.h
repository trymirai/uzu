

using namespace steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
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
METAL_FUNC void gemm_splitk_impl(
    const device T* a,
    const device T* b,
    device U* c,
    const constant GEMMSpiltKParams* params,
    threadgroup T* a_shared,
    threadgroup T* b_shared,
    uint simd_lane_id,
    uint simd_group_id,
    uint3 tid,
    uint3 lid
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

  const int tid_x = tid.x;
  const int tid_y = tid.y;
  const int tid_z = tid.z;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  // Find block in a, b, c
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int k_start = params->split_k_partition_size * tid_z;

  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);
  const size_t k_start_long = size_t(k_start);

  a += transpose_a ? (c_row_long + k_start_long * params->lda)
                   : (k_start_long + c_row_long * params->lda);
  b += transpose_b ? (k_start_long + c_col_long * params->ldb)
                   : (c_col_long + k_start_long * params->ldb);
  c += (size_t(params->split_k_partition_stride) * tid_z) +
       (c_row_long * params->ldc + c_col_long);

  // Prepare threadgroup loading operations
  thread loader_a_t loader_a(a, params->lda, a_shared, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(b, params->ldb, b_shared, simd_group_id, simd_lane_id);

  // Prepare threadgroup mma operation
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  short tgp_bm = min(BM, params->M - c_row);
  short tgp_bn = min(BN, params->N - c_col);
  short leftover_bk = params->K % BK;

  if (MN_aligned || (tgp_bm == BM && tgp_bn == BN)) {
    gemm_kernel::gemm_loop(
        a_shared,
        b_shared,
        gemm_k_iterations,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        leftover_bk,
        LoopAlignment<true, true, true>{}
    );
  } else if (tgp_bn == BN) {
    gemm_kernel::gemm_loop(
        a_shared,
        b_shared,
        gemm_k_iterations,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        leftover_bk,
        LoopAlignment<false, true, true>{}
    );
  } else if (tgp_bm == BM) {
    gemm_kernel::gemm_loop(
        a_shared,
        b_shared,
        gemm_k_iterations,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        leftover_bk,
        LoopAlignment<true, false, true>{}
    );
  } else {
    gemm_kernel::gemm_loop(
        a_shared,
        b_shared,
        gemm_k_iterations,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        leftover_bk,
        LoopAlignment<false, false, true>{}
    );
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if ((tid_z + 1) == (params->split_k_partitions)) {
    int gemm_k_iter_remaining =
        (params->K - (k_start + params->split_k_partition_size)) / BK;
    if (!K_aligned || gemm_k_iter_remaining > 0)
      gemm_kernel::gemm_loop(
          a_shared,
          b_shared,
          gemm_k_iter_remaining,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<false, false, K_aligned>{}
      );
  }

  if (MN_aligned || (tgp_bm == BM && tgp_bn == BN)) {
    mma_op.store_result(c, params->ldc);
  } else {
    mma_op.store_result_safe(c, params->ldc, short2(tgp_bn, tgp_bm));
  }
}

///////////////////////////////////////////////////////////////////////////////
// Split k accumulation kernel
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
  // Ajust d and c
  d += gid.x + gid.y * size_t(ldd);
  c_split += gid.x + gid.y * size_t(ldd);

  size_t offset = 0;
  AccT out = 0;

  for (int i = 0; i < k_partitions; i++) {
    out += c_split[offset];
    offset += partition_stride;
  }

  // Write output
  d[0] = Epilogue::apply(out);
}

