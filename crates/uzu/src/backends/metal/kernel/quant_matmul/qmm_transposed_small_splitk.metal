#include <metal_stdlib>
#include "../common/dsl.h"
#include "quant_matmul.h"

// Split-K variant of QmmTransposedSmall.
//
// Pass 1 (partial GEMM): each TG processes K_chunk = K/split_k elements.
//   Grid Z = split_k.  Output: partial[split_k * M * N] stored as T (bfloat).
//   Using the float accumulator for the reduction, via T intermediate.
//
// Pass 2 (reduction): sums split_k T partial tiles → final T output in float.
//   Grid Z = 1.
//
// BM=8, BK=32, BN=32, WM=1, WN=1 → 1 simdgroup, 32 threads.

// ---------------------------------------------------------------------------
// Impl helper: partial GEMM over K range [k_start, k_start+k_chunk).
//   full_k: the full K dimension (used for weight/scale stride).
//   k_chunk: size of this TG's K slice (must be divisible by BK=32).
// ---------------------------------------------------------------------------

template <
    typename T,
    const int GROUP_SIZE,
    const int BITS,
    const bool use_mlx_quant>
inline void qmm_transposed_small_splitk_impl(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* x,
    device T* partial_slice,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const int full_k,
    const int k_chunk,
    const int k_start,
    const int n,
    const int m,
    uint3 tid,
    uint simd_gid,
    uint simd_lid
) {
  constexpr int BM = 8;
  constexpr int BK = 32;
  constexpr int BN = 32;
  constexpr int WM = 1;
  constexpr int WN = 1;
  constexpr int pack_factor = get_pack_factor<BITS, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<BITS>();
  constexpr int BK_padded = BK + 16 / sizeof(T);

  // full-K strides (for weight and scale pointer arithmetic)
  const int K_w_full = full_k * bytes_per_pack / pack_factor;
  const int K_g_full = (full_k + GROUP_SIZE - 1) / GROUP_SIZE;

  const int y_row = (int)tid.y * BM;
  const int y_col = (int)tid.x * BN;

  if (y_row >= m || y_col >= n) {
    return;
  }

  auto wl = (const device uint8_t*)w;

  // Adjust x to row y_row, column k_start
  const device T* x_block = x + y_row * (int64_t)full_k + k_start;
  // Adjust w to output column y_col, row k_start
  const device uint8_t* w_block =
      wl + (int64_t)y_col * K_w_full +
      (int64_t)k_start * bytes_per_pack / pack_factor;

  device T* y_block = partial_slice + y_row * n + y_col;

  const short num_els = (short)min(BM, m - y_row);
  const short num_outs = (short)min(BN, n - y_col);

  using mma_t = matmul_utils::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      matmul_utils::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * 32>;

  loader_x_t loader_x(x_block, full_k, Xs, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (use_mlx_quant) {
    // scales/biases: [n_out_rows, K_g_full], advance to output col + k_start
    // group
    const device T* scales_block =
        scales + (int64_t)y_col * K_g_full + k_start / GROUP_SIZE;
    const device T* biases_block =
        biases + (int64_t)y_col * K_g_full + k_start / GROUP_SIZE;

    using loader_w_t = QuantizedBlockLoaderMlx<
        T,
        BN,
        BK,
        BK_padded,
        1,
        WM * WN * 32,
        GROUP_SIZE,
        BITS>;

    loader_w_t loader_w(
        w_block,
        scales_block,
        biases_block,
        full_k,
        Ws,
        simd_gid,
        simd_lid
    );

    // Loop over k_chunk in BK steps — always full tiles (k_chunk divisible by
    // BK)
    const int n_tiles = k_chunk / BK;
    for (int t = 0; t < n_tiles; t++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (num_els < BM) {
        loader_x.load_safe(short2(BK, num_els));
      } else {
        loader_x.load_unsafe();
      }
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  } else {
    const int K_g_chunk = k_chunk / GROUP_SIZE;
    const int zp_stride_full = (BITS == 4) ? ((K_g_full + 1) / 2) : K_g_full;
    // zero_points: [n_out_rows, zp_stride_full], advance to output col + k
    // group
    const device uint8_t* zp_block =
        zero_points + (int64_t)y_col * zp_stride_full +
        ((BITS == 4) ? (k_start / GROUP_SIZE / 2) : (k_start / GROUP_SIZE));
    const device T* scales_block =
        scales + (int64_t)y_col * K_g_full + k_start / GROUP_SIZE;

    using loader_w_t = QuantizedBlockLoaderZp<
        T,
        BN,
        BK,
        BK_padded,
        1,
        WM * WN * 32,
        GROUP_SIZE,
        BITS>;

    loader_w_t loader_w(
        w_block,
        scales_block,
        zp_block,
        full_k,
        K_g_chunk,
        Ws,
        simd_gid,
        simd_lid
    );

    const int n_tiles = k_chunk / BK;
    for (int t = 0; t < n_tiles; t++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (num_els < BM) {
        loader_x.load_safe(short2(BK, num_els));
      } else {
        loader_x.load_unsafe();
      }
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  }

  // Store result
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y_block, n, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y_block, n);
  }
}

// ---------------------------------------------------------------------------
// Pass 1 – partial GEMM kernel
// ---------------------------------------------------------------------------

template <typename T, int GROUP_SIZE, int BITS>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4, 8)
PUBLIC KERNEL(QuantizedMatmulQmmTransposedSmallSplitKPartial)(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device T* x,
    device T* partial,
    const constant int& k,
    const constant int& n,
    const constant int& m,
    const constant int& split_k,
    threadgroup T Xs[8 * (32 + 16 / sizeof(T))],
    threadgroup T Ws[32 * (32 + 16 / sizeof(T))],
    const bool use_zero_points SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    const uint tgid_x GROUPS((n + 32 - 1) / 32),
    const uint tgid_y GROUPS((m + 8 - 1) / 8),
    const uint tgid_z GROUPS(split_k),
    const uint tid_x THREADS(32),
    const uint tid_y THREADS(1),
    const uint tid_z THREADS(1)
) {
  const int k_chunk = k / split_k;
  const int k_start = (int)tgid_z * k_chunk;

  const uint3 tile_tid = uint3(tgid_x, tgid_y, 0);
  const uint simd_lid = tid_x;
  const uint simd_gid = 0;

  // Each split-k slice: [M, N] block at offset tgid_z * m * n
  device T* partial_slice = partial + (int)tgid_z * m * n;

  if (use_mlx_quant) {
    qmm_transposed_small_splitk_impl<T, GROUP_SIZE, BITS, true>(
        w,
        scales,
        zero_points,
        biases,
        x,
        partial_slice,
        Xs,
        Ws,
        k,
        k_chunk,
        k_start,
        n,
        m,
        tile_tid,
        simd_gid,
        simd_lid
    );
  } else {
    qmm_transposed_small_splitk_impl<T, GROUP_SIZE, BITS, false>(
        w,
        scales,
        zero_points,
        biases,
        x,
        partial_slice,
        Xs,
        Ws,
        k,
        k_chunk,
        k_start,
        n,
        m,
        tile_tid,
        simd_gid,
        simd_lid
    );
  }
}

// ---------------------------------------------------------------------------
// Pass 2 – reduction: sum split_k partial T slices → final T output
// ---------------------------------------------------------------------------

template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(QuantizedMatmulQmmTransposedSmallSplitKReduce)(
    const device T* partial,
    device T* y,
    const constant int& n,
    const constant int& m,
    const constant int& split_k,
    const uint tgid_x GROUPS((n + 32 - 1) / 32),
    const uint tgid_y GROUPS((m + 8 - 1) / 8),
    const uint tgid_z GROUPS(1),
    const uint tid_x THREADS(32),
    const uint tid_y THREADS(8),
    const uint tid_z THREADS(1)
) {
  const int row = (int)tgid_y * 8 + (int)tid_y;
  const int col = (int)tgid_x * 32 + (int)tid_x;

  if (row >= m || col >= n) {
    return;
  }

  float sum = 0.0f;
  const int slice_size = m * n;
  for (int s = 0; s < split_k; s++) {
    sum += (float)partial[s * slice_size + row * n + col];
  }

  y[row * n + col] = (T)sum;
}
