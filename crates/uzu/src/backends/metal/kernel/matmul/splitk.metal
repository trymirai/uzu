// Split-K GEMM partial accumulation kernels (simplified).
// Partial products are accumulated into a float buffer; a second kernel
// converts to the target type.

#include <metal_stdlib>
using namespace metal;

template <typename T, typename AccT>
METAL_FUNC void splitk_partial_impl(
    const device T* __restrict a [[buffer(0)]],
    const device T* __restrict b [[buffer(1)]],
    device atomic_float* __restrict accum [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& lda [[buffer(6)]],
    constant uint& ldb [[buffer(7)]],
    constant uint& ldd [[buffer(8)]],
    constant uint& k_start [[buffer(9)]],
    constant uint& k_end [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    const uint col = tgid.x * 8 + tid.x;
    const uint row = tgid.y * 8 + tid.y;
    const uint batch = tgid.z;

    if (row >= m || col >= n) {
        return;
    }

    const device T* a_row = a + batch * (lda * m) + row * lda;
    const device T* b_col = b + col * ldb;

    AccT acc = static_cast<AccT>(0);
    for (uint kk = k_start; kk < k_end; kk++) {
        acc += static_cast<AccT>(a_row[kk]) *
            static_cast<AccT>(b_col[kk]);
    }

    const uint out_index = batch * (ldd * m) + row * ldd + col;
    atomic_fetch_add_explicit(&accum[out_index], acc, memory_order_relaxed);
}

template <typename T>
METAL_FUNC void splitk_convert_impl(
    const device atomic_float* __restrict accum [[buffer(0)]],
    device T* __restrict out [[buffer(1)]],
    constant uint& m [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& ldd [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    const uint col = tgid.x * 8 + tid.x;
    const uint row = tgid.y * 8 + tid.y;
    const uint batch = tgid.z;

    if (row >= m || col >= n) {
        return;
    }

    const uint idx = batch * (ldd * m) + row * ldd + col;
    const float v = atomic_load_explicit(&accum[idx], memory_order_relaxed);
    out[idx] = static_cast<T>(v);
}

// Kernel entry points
[[kernel, max_total_threads_per_threadgroup(256)]] void splitk_partial_f16(
    const device half* a [[buffer(0)]],
    const device half* b [[buffer(1)]],
    device atomic_float* accum [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& lda [[buffer(6)]],
    constant uint& ldb [[buffer(7)]],
    constant uint& ldd [[buffer(8)]],
    constant uint& k_start [[buffer(9)]],
    constant uint& k_end [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    splitk_partial_impl<half, float>(
        a, b, accum, m, n, k, lda, ldb, ldd, k_start, k_end, tgid, tid);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void splitk_partial_bf16(
    const device bfloat* a [[buffer(0)]],
    const device bfloat* b [[buffer(1)]],
    device atomic_float* accum [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& lda [[buffer(6)]],
    constant uint& ldb [[buffer(7)]],
    constant uint& ldd [[buffer(8)]],
    constant uint& k_start [[buffer(9)]],
    constant uint& k_end [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    splitk_partial_impl<bfloat, float>(
        a, b, accum, m, n, k, lda, ldb, ldd, k_start, k_end, tgid, tid);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void splitk_convert_f16(
    const device atomic_float* accum [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& m [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& ldd [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    splitk_convert_impl<half>(accum, out, m, n, ldd, tgid, tid);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void splitk_convert_bf16(
    const device atomic_float* accum [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant uint& m [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& ldd [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    splitk_convert_impl<bfloat>(accum, out, m, n, ldd, tgid, tid);
}


