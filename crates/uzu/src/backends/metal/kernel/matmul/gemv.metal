// SPDX-License-Identifier: MIT

#include <metal_stdlib>
using namespace metal;

template <typename T, typename AccT, int ROWS_PER_SIMDGROUP, int ELEMS_PER_THREAD>
METAL_FUNC void gemv_multirow(
    const device T* __restrict weights,
    const device T* __restrict input,
    device T* __restrict output,
    const int input_dim,
    const int output_dim,
    const int weight_stride,
    const int batch_idx,
    const int input_batch_stride,
    const int output_batch_stride,
    uint row_base,
    uint simd_lid) {

    constexpr int BLOCK_K = 32 * ELEMS_PER_THREAD;

    input += batch_idx * input_batch_stride;
    output += batch_idx * output_batch_stride;

    if (row_base >= static_cast<uint>(output_dim)) {
        return;
    }

    const int rows_to_compute = min(ROWS_PER_SIMDGROUP, output_dim - static_cast<int>(row_base));

    AccT accumulators[ROWS_PER_SIMDGROUP];
    #pragma unroll(ROWS_PER_SIMDGROUP)
    for (int r = 0; r < ROWS_PER_SIMDGROUP; ++r) {
        accumulators[r] = AccT(0);
    }

    AccT input_cache[ELEMS_PER_THREAD];

    const device T* weight_row_ptrs[ROWS_PER_SIMDGROUP];
    #pragma unroll(ROWS_PER_SIMDGROUP)
    for (int r = 0; r < ROWS_PER_SIMDGROUP; ++r) {
        weight_row_ptrs[r] = weights + (row_base + r) * weight_stride;
    }

    int k = 0;
    for (; k + BLOCK_K <= input_dim; k += BLOCK_K) {
        const int thread_k = k + simd_lid * ELEMS_PER_THREAD;

        #pragma unroll(ELEMS_PER_THREAD)
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            input_cache[e] = static_cast<AccT>(input[thread_k + e]);
        }

        #pragma unroll(ROWS_PER_SIMDGROUP)
        for (int r = 0; r < ROWS_PER_SIMDGROUP; ++r) {
            AccT dot = 0;
            #pragma unroll(ELEMS_PER_THREAD)
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                dot += input_cache[e] * static_cast<AccT>(weight_row_ptrs[r][thread_k + e]);
            }
            accumulators[r] += dot;
        }
    }

    if (k < input_dim) {
        const int thread_k = k + simd_lid * ELEMS_PER_THREAD;
        const int remaining = input_dim - k;

        #pragma unroll(ELEMS_PER_THREAD)
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            const int idx = simd_lid * ELEMS_PER_THREAD + e;
            input_cache[e] = (idx < remaining) ? static_cast<AccT>(input[thread_k + e]) : AccT(0);
        }

        #pragma unroll(ROWS_PER_SIMDGROUP)
        for (int r = 0; r < ROWS_PER_SIMDGROUP; ++r) {
            AccT dot = 0;
            #pragma unroll(ELEMS_PER_THREAD)
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                const int idx = simd_lid * ELEMS_PER_THREAD + e;
                if (idx < remaining) {
                    dot += input_cache[e] * static_cast<AccT>(weight_row_ptrs[r][thread_k + e]);
                }
            }
            accumulators[r] += dot;
        }
    }

    #pragma unroll(ROWS_PER_SIMDGROUP)
    for (int r = 0; r < ROWS_PER_SIMDGROUP; ++r) {
        accumulators[r] = simd_sum(accumulators[r]);
    }

    if (simd_lid == 0) {
        #pragma unroll(ROWS_PER_SIMDGROUP)
        for (int r = 0; r < rows_to_compute; ++r) {
            output[row_base + r] = static_cast<T>(accumulators[r]);
        }
    }
}

#define DECL_GEMV_FAST(NAME, TYPE, ACC)                                              \
    [[kernel]] void NAME(                                                            \
        const device TYPE* weights [[buffer(0)]],                                    \
        const device TYPE* input [[buffer(1)]],                                      \
        device TYPE* output [[buffer(2)]],                                           \
        constant int& input_dim [[buffer(3)]],                                       \
        constant int& output_dim [[buffer(4)]],                                      \
        constant int& weight_stride [[buffer(5)]],                                   \
        constant int& input_batch_stride [[buffer(6)]],                              \
        constant int& output_batch_stride [[buffer(7)]],                             \
        uint3 tgid [[threadgroup_position_in_grid]],                                 \
        uint simd_lid [[thread_index_in_simdgroup]]) {                               \
        constexpr int ROWS_PER_SIMDGROUP = 4;                                        \
        constexpr int ELEMS_PER_THREAD = 4;                                          \
        const uint row_base = tgid.x * ROWS_PER_SIMDGROUP;                           \
        gemv_multirow<TYPE, ACC, ROWS_PER_SIMDGROUP, ELEMS_PER_THREAD>(              \
            weights, input, output, input_dim, output_dim, weight_stride,            \
            tgid.z, input_batch_stride, output_batch_stride, row_base, simd_lid);    \
    }

DECL_GEMV_FAST(gemv_fast_f16, half, float)
DECL_GEMV_FAST(gemv_fast_bf16, bfloat, float)
DECL_GEMV_FAST(gemv_fast_f32, float, float)

template <typename T, typename AccT, ushort ROWS_PER_THREADGROUP, ushort TILE_K>
METAL_FUNC void gemv_tiled(
    const device T* __restrict weights,
    const device T* __restrict input,
    device T* __restrict output,
    constant uint& input_dim,
    constant uint& output_dim,
    constant uint& weight_stride,
    constant uint& input_stride,
    constant uint& output_stride,
    uint3 tgid,
    uint simd_gid,
    uint simd_lid,
    uint tid,
    threadgroup AccT* input_tile) {

    const ushort THREADS_PER_THREADGROUP = ROWS_PER_THREADGROUP * 32;
    if (simd_gid >= ROWS_PER_THREADGROUP) {
        return;
    }

    const uint row = tgid.x * ROWS_PER_THREADGROUP + simd_gid;
    if (row >= output_dim) {
        return;
    }

    const uint batch = tgid.z;
    const device T* weight_row = weights + row * weight_stride;
    const device T* input_ptr = input + batch * input_stride;
    device T* output_ptr = output + batch * output_stride;

    AccT accumulator = AccT(0);

    for (uint k_block = 0; k_block < input_dim; k_block += TILE_K) {
        const uint remaining = input_dim - k_block;
        const uint tile_size = remaining < TILE_K ? remaining : TILE_K;

        if (tid < tile_size && tid < THREADS_PER_THREADGROUP) {
            input_tile[tid] = static_cast<AccT>(input_ptr[k_block + tid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = simd_lid; idx < tile_size; idx += 32) {
            accumulator += static_cast<AccT>(weight_row[k_block + idx]) * input_tile[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (ushort offset = 16; offset > 0; offset >>= 1) {
        accumulator += simd_shuffle_down(accumulator, offset);
    }

    if (simd_lid == 0) {
        output_ptr[row] = static_cast<T>(accumulator);
    }
}

#define DECL_GEMV_TILED(NAME, TYPE, ACC, ROWS, TILE_K)                               \
    [[kernel, max_total_threads_per_threadgroup(ROWS * 32)]] void NAME(              \
        const device TYPE* weights [[buffer(0)]],                                    \
        const device TYPE* input [[buffer(1)]],                                      \
        device TYPE* output [[buffer(2)]],                                           \
        constant uint& input_dim [[buffer(3)]],                                      \
        constant uint& output_dim [[buffer(4)]],                                     \
        constant uint& weight_stride [[buffer(5)]],                                  \
        constant uint& input_stride [[buffer(6)]],                                   \
        constant uint& output_stride [[buffer(7)]],                                  \
        uint3 tgid [[threadgroup_position_in_grid]],                                 \
        uint simd_gid [[simdgroup_index_in_threadgroup]],                            \
        uint simd_lid [[thread_index_in_simdgroup]],                                 \
        uint tid [[thread_index_in_threadgroup]]) {                                  \
        threadgroup ACC input_tile[TILE_K];                                          \
        gemv_tiled<TYPE, ACC, ROWS, TILE_K>(                                         \
            weights, input, output, input_dim, output_dim, weight_stride,            \
            input_stride, output_stride, tgid, simd_gid, simd_lid, tid, input_tile); \
    }

DECL_GEMV_TILED(gemv_f16_rows2, half, float, 2, 128)
DECL_GEMV_TILED(gemv_f16_rows4, half, float, 4, 128)
DECL_GEMV_TILED(gemv_f16_rows8, half, float, 8, 128)

DECL_GEMV_TILED(gemv_bf16_rows2, bfloat, float, 2, 128)
DECL_GEMV_TILED(gemv_bf16_rows4, bfloat, float, 4, 128)
DECL_GEMV_TILED(gemv_bf16_rows8, bfloat, float, 8, 128)
