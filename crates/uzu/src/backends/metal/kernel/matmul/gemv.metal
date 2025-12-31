// SPDX-License-Identifier: MIT
// Optimized GEMV kernels:
// - Multi-row output per simdgroup with input vector reuse
// - Vectorized loads (4 elements per thread)
// - simd_sum for fast K-dimension reduction
// - Register-only (no threadgroup memory)
//
// Two variants:
// - gemv_fast_*: 4 rows per simdgroup, best for large K (≥4096)
// - gemv_wide_*: 16 rows per simdgroup, best for small K with large N (MLP up pattern)

#include <metal_stdlib>
using namespace metal;

// Elements loaded per thread per K iteration (4 elements vectorized)
#define ELEMS_PER_THREAD 4

// Templated GEMV implementation with configurable rows per simdgroup
// y = A * x where A is [N, K] row-major
template <typename T, typename AccT, int ROWS>
METAL_FUNC void gemv_impl(
    const device T* __restrict A,
    const device T* __restrict x,
    device T* __restrict y,
    const int K,
    const int N,
    const int lda,
    const int batch_idx,
    const int batch_stride_x,
    const int batch_stride_y,
    uint row_base,
    uint simd_lid) {

    // Block size in K dimension: 32 threads * 4 elements = 128
    constexpr int BLOCK_K = 32 * ELEMS_PER_THREAD;

    // Position pointers for this batch
    x += batch_idx * batch_stride_x;
    y += batch_idx * batch_stride_y;

    // Bounds check
    if (row_base >= static_cast<uint>(N)) {
        return;
    }

    const int rows_to_compute = min(ROWS, N - static_cast<int>(row_base));

    // Accumulators in registers
    AccT result[ROWS];
    #pragma unroll(ROWS)
    for (int r = 0; r < ROWS; ++r) {
        result[r] = AccT(0);
    }

    // Input vector cache
    AccT x_cache[ELEMS_PER_THREAD];

    // Weight row pointers
    const device T* row_ptrs[ROWS];
    #pragma unroll(ROWS)
    for (int r = 0; r < ROWS; ++r) {
        row_ptrs[r] = A + (row_base + r) * lda;
    }

    // Main K loop
    int k = 0;
    for (; k + BLOCK_K <= K; k += BLOCK_K) {
        const int thread_k = k + simd_lid * ELEMS_PER_THREAD;

        // Load input vector chunk
        #pragma unroll(ELEMS_PER_THREAD)
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            x_cache[e] = static_cast<AccT>(x[thread_k + e]);
        }

        // Dot product for each row
        #pragma unroll(ROWS)
        for (int r = 0; r < ROWS; ++r) {
            AccT dot = 0;
            #pragma unroll(ELEMS_PER_THREAD)
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                dot += x_cache[e] * static_cast<AccT>(row_ptrs[r][thread_k + e]);
            }
            result[r] += dot;
        }
    }

    // Handle leftover K elements
    if (k < K) {
        const int thread_k = k + simd_lid * ELEMS_PER_THREAD;
        const int remaining = K - k;

        #pragma unroll(ELEMS_PER_THREAD)
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            const int idx = simd_lid * ELEMS_PER_THREAD + e;
            x_cache[e] = (idx < remaining) ? static_cast<AccT>(x[thread_k + e]) : AccT(0);
        }

        #pragma unroll(ROWS)
        for (int r = 0; r < ROWS; ++r) {
            AccT dot = 0;
            #pragma unroll(ELEMS_PER_THREAD)
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                const int idx = simd_lid * ELEMS_PER_THREAD + e;
                if (idx < remaining) {
                    dot += x_cache[e] * static_cast<AccT>(row_ptrs[r][thread_k + e]);
                }
            }
            result[r] += dot;
        }
    }

    // Reduce across simdgroup
    #pragma unroll(ROWS)
    for (int r = 0; r < ROWS; ++r) {
        result[r] = simd_sum(result[r]);
    }

    // Write output
    if (simd_lid == 0) {
        #pragma unroll(ROWS)
        for (int r = 0; r < rows_to_compute; ++r) {
            y[row_base + r] = static_cast<T>(result[r]);
        }
    }
}

// Standard GEMV: 4 rows per simdgroup (best for large K)
#define DECL_GEMV_FAST(NAME, TYPE, ACC)                                         \
    [[kernel]] void NAME(                                                       \
        const device TYPE* A [[buffer(0)]],                                     \
        const device TYPE* x [[buffer(1)]],                                     \
        device TYPE* y [[buffer(2)]],                                           \
        constant int& K [[buffer(3)]],                                          \
        constant int& N [[buffer(4)]],                                          \
        constant int& lda [[buffer(5)]],                                        \
        constant int& batch_stride_x [[buffer(6)]],                             \
        constant int& batch_stride_y [[buffer(7)]],                             \
        uint3 tgid [[threadgroup_position_in_grid]],                            \
        uint simd_lid [[thread_index_in_simdgroup]]) {                          \
        constexpr int ROWS = 4;                                                 \
        const uint row_base = tgid.x * ROWS;                                    \
        gemv_impl<TYPE, ACC, ROWS>(                                             \
            A, x, y, K, N, lda, tgid.z, batch_stride_x, batch_stride_y,         \
            row_base, simd_lid);                                                \
    }

// Wide GEMV: 8 rows per simdgroup (balanced for small K, large N)
#define DECL_GEMV_WIDE(NAME, TYPE, ACC)                                         \
    [[kernel]] void NAME(                                                       \
        const device TYPE* A [[buffer(0)]],                                     \
        const device TYPE* x [[buffer(1)]],                                     \
        device TYPE* y [[buffer(2)]],                                           \
        constant int& K [[buffer(3)]],                                          \
        constant int& N [[buffer(4)]],                                          \
        constant int& lda [[buffer(5)]],                                        \
        constant int& batch_stride_x [[buffer(6)]],                             \
        constant int& batch_stride_y [[buffer(7)]],                             \
        uint3 tgid [[threadgroup_position_in_grid]],                            \
        uint simd_lid [[thread_index_in_simdgroup]]) {                          \
        constexpr int ROWS = 8;                                                 \
        const uint row_base = tgid.x * ROWS;                                    \
        gemv_impl<TYPE, ACC, ROWS>(                                             \
            A, x, y, K, N, lda, tgid.z, batch_stride_x, batch_stride_y,         \
            row_base, simd_lid);                                                \
    }

// Instantiate standard (4-row) kernels
DECL_GEMV_FAST(gemv_fast_f16, half, float)
DECL_GEMV_FAST(gemv_fast_bf16, bfloat, float)
DECL_GEMV_FAST(gemv_fast_f32, float, float)

// Instantiate wide (8-row) kernels
DECL_GEMV_WIDE(gemv_wide_f16, half, float)
DECL_GEMV_WIDE(gemv_wide_bf16, bfloat, float)
DECL_GEMV_WIDE(gemv_wide_f32, float, float)

// =============================================================================
// Parallel GEMV: Multiple simdgroups per threadgroup with shared input vector
// Best for small K with large N (reduces threadgroup count, amortizes overhead)
// =============================================================================

template <typename T, typename AccT, int SIMDGROUPS_PER_TG, int BK>
[[kernel, max_total_threads_per_threadgroup(SIMDGROUPS_PER_TG * 32)]]
void gemv_parallel_impl(
    const device T* __restrict A [[buffer(0)]],
    const device T* __restrict x [[buffer(1)]],
    device T* __restrict y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& lda [[buffer(5)]],
    constant int& batch_stride_x [[buffer(6)]],
    constant int& batch_stride_y [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {

    // Shared memory for input vector tile
    threadgroup AccT x_shared[BK];

    // Position pointers for this batch
    const int batch_idx = tgid.z;
    x += batch_idx * batch_stride_x;
    y += batch_idx * batch_stride_y;

    // Each simdgroup computes one output row
    const uint row = tgid.x * SIMDGROUPS_PER_TG + simd_gid;
    if (row >= static_cast<uint>(N)) {
        return;
    }

    const device T* row_ptr = A + row * lda;
    AccT acc = AccT(0);

    // Process K in tiles
    for (int k_start = 0; k_start < K; k_start += BK) {
        const int remaining = K - k_start;
        const int tile_k = remaining < BK ? remaining : BK;

        // Cooperative load of input vector tile into shared memory
        for (int idx = tid; idx < tile_k; idx += SIMDGROUPS_PER_TG * 32) {
            x_shared[idx] = static_cast<AccT>(x[k_start + idx]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread in simdgroup processes part of the tile
        for (int idx = simd_lid; idx < tile_k; idx += 32) {
            acc += static_cast<AccT>(row_ptr[k_start + idx]) * x_shared[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce within simdgroup
    acc = simd_sum(acc);

    // Write output
    if (simd_lid == 0) {
        y[row] = static_cast<T>(acc);
    }
}

// Parallel kernels: 32 simdgroups per threadgroup, BK=1024 for maximum occupancy
template [[host_name("gemv_parallel_f16")]] kernel void gemv_parallel_impl<half, float, 32, 1024>(
    const device half*, const device half*, device half*,
    constant int&, constant int&, constant int&, constant int&, constant int&,
    uint3, uint, uint, uint);
template [[host_name("gemv_parallel_bf16")]] kernel void gemv_parallel_impl<bfloat, float, 32, 1024>(
    const device bfloat*, const device bfloat*, device bfloat*,
    constant int&, constant int&, constant int&, constant int&, constant int&,
    uint3, uint, uint, uint);
template [[host_name("gemv_parallel_f32")]] kernel void gemv_parallel_impl<float, float, 32, 1024>(
    const device float*, const device float*, device float*,
    constant int&, constant int&, constant int&, constant int&, constant int&,
    uint3, uint, uint, uint);

// =============================================================================
// Legacy kernels for backwards compatibility (used by existing Rust dispatcher)
// =============================================================================

template <typename T, typename AccT, ushort ROWS_PER_TG, ushort BK>
METAL_FUNC void gemv_impl_tiled(
    const device T* __restrict matrix [[buffer(0)]],
    const device T* __restrict vector [[buffer(1)]],
    device T* __restrict output [[buffer(2)]],
    constant uint& k [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& ldb [[buffer(5)]],
    constant uint& lda [[buffer(6)]],
    constant uint& ldd [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup AccT* vec_tile) {
    const ushort TG_THREADS = ROWS_PER_TG * 32;
    if (simd_gid >= ROWS_PER_TG) {
        return;
    }
    const uint row = tgid.x * ROWS_PER_TG + simd_gid;
    if (row >= n) {
        return;
    }
    const uint batch = tgid.z;

    const device T* row_ptr = matrix + row * ldb;
    const device T* vec_ptr = vector + batch * lda;
    device T* out_ptr = output + batch * ldd;

    AccT acc = static_cast<AccT>(0);

    for (uint k_block = 0; k_block < k; k_block += BK) {
        const uint remaining = k - k_block;
        const uint tile_elems = remaining < BK ? remaining : BK;

        if (tid < tile_elems && tid < TG_THREADS) {
            vec_tile[tid] = static_cast<AccT>(vec_ptr[k_block + tid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = simd_lid; idx < tile_elems; idx += 32) {
            acc += static_cast<AccT>(row_ptr[k_block + idx]) * vec_tile[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (ushort offset = 16; offset > 0; offset >>= 1) {
        acc += simd_shuffle_down(acc, offset);
    }

    if (simd_lid == 0) {
        out_ptr[row] = static_cast<T>(acc);
    }
}

#define DECL_GEMV_LEGACY(NAME, TYPE, ACC, ROWS, BK)                             \
    [[kernel, max_total_threads_per_threadgroup(ROWS * 32)]] void NAME(         \
        const device TYPE* matrix [[buffer(0)]],                                \
        const device TYPE* vector [[buffer(1)]],                                \
        device TYPE* output [[buffer(2)]],                                      \
        constant uint& k [[buffer(3)]],                                         \
        constant uint& n [[buffer(4)]],                                         \
        constant uint& ldb [[buffer(5)]],                                       \
        constant uint& lda [[buffer(6)]],                                       \
        constant uint& ldd [[buffer(7)]],                                       \
        uint3 tgid [[threadgroup_position_in_grid]],                            \
        uint simd_gid [[simdgroup_index_in_threadgroup]],                       \
        uint simd_lid [[thread_index_in_simdgroup]],                            \
        uint tid [[thread_index_in_threadgroup]]) {                             \
        threadgroup ACC vec_tile[BK];                                           \
        gemv_impl_tiled<TYPE, ACC, ROWS, BK>(                                   \
            matrix, vector, output, k, n, ldb, lda, ldd,                        \
            tgid, simd_gid, simd_lid, tid, vec_tile);                           \
    }

// Legacy half variants
DECL_GEMV_LEGACY(gemv_f16_rows2, half, float, 2, 128)
DECL_GEMV_LEGACY(gemv_f16_rows4, half, float, 4, 128)
DECL_GEMV_LEGACY(gemv_f16_rows8, half, float, 8, 128)

// Legacy bfloat16 variants
DECL_GEMV_LEGACY(gemv_bf16_rows2, bfloat, float, 2, 128)
DECL_GEMV_LEGACY(gemv_bf16_rows4, bfloat, float, 4, 128)
DECL_GEMV_LEGACY(gemv_bf16_rows8, bfloat, float, 8, 128)
