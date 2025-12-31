// Masked and gather GEMV variants. Masks/indices are honored only for output
// write; data loads remain dense.

#include <metal_stdlib>
using namespace metal;

template <typename T, typename AccT, ushort ROWS>
METAL_FUNC void gemv_masked_impl(
    const device T* __restrict matrix [[buffer(0)]],
    const device T* __restrict vector [[buffer(1)]],
    const device uchar* __restrict out_mask [[buffer(2)]],
    const device uchar* __restrict op_mask [[buffer(3)]],
    device T* __restrict output [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& ldb [[buffer(7)]],
    constant uint& lda [[buffer(8)]],
    constant uint& ldd [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    if (simd_gid >= ROWS) {
        return;
    }
    const uint row = tgid.x * ROWS + simd_gid;
    if (row >= n) {
        return;
    }
    const uint batch = tgid.z;
    const device T* row_ptr = matrix + row * ldb;
    const device T* vec_ptr = vector + batch * lda;
    device T* out_ptr = output + batch * ldd;

    AccT acc = static_cast<AccT>(0);
    for (uint idx = simd_lid; idx < k; idx += 32) {
        // If op_mask provided, zero out masked elements
        const bool op_on = (op_mask == nullptr) || (op_mask[row] != 0);
        if (op_on) {
            acc += static_cast<AccT>(row_ptr[idx]) *
                static_cast<AccT>(vec_ptr[idx]);
        }
    }

    for (ushort offset = 16; offset > 0; offset >>= 1) {
        acc += simd_shuffle_down(acc, offset);
    }

    if (simd_lid == 0) {
        if (out_mask == nullptr || out_mask[row] != 0) {
            out_ptr[row] = static_cast<T>(acc);
        }
    }
}

template <typename T, typename AccT, ushort ROWS>
METAL_FUNC void gemv_gather_impl(
    const device T* __restrict matrix [[buffer(0)]],
    const device T* __restrict vector [[buffer(1)]],
    const device uint* __restrict vec_indices [[buffer(2)]],
    const device uint* __restrict mat_indices [[buffer(3)]],
    device T* __restrict output [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& ldb [[buffer(7)]],
    constant uint& lda [[buffer(8)]],
    constant uint& ldd [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    if (simd_gid >= ROWS) {
        return;
    }
    const uint row = tgid.x * ROWS + simd_gid;
    if (row >= n) {
        return;
    }
    const uint batch = tgid.z;
    const device T* base_row = matrix + row * ldb;
    device T* out_ptr = output + batch * ldd;

    AccT acc = static_cast<AccT>(0);
    for (uint idx = simd_lid; idx < k; idx += 32) {
        const uint col = mat_indices ? mat_indices[idx] : idx;
        const uint vec_idx = vec_indices ? vec_indices[idx] : idx;
        acc += static_cast<AccT>(base_row[col]) *
            static_cast<AccT>(vector[batch * lda + vec_idx]);
    }

    for (ushort offset = 16; offset > 0; offset >>= 1) {
        acc += simd_shuffle_down(acc, offset);
    }

    if (simd_lid == 0) {
        out_ptr[row] = static_cast<T>(acc);
    }
}

#define DECL_GEMV_MASKED(NAME, TYPE, ACC, ROWS)                               \
    [[kernel, max_total_threads_per_threadgroup(ROWS * 32)]] void NAME(       \
        const device TYPE* matrix [[buffer(0)]],                              \
        const device TYPE* vector [[buffer(1)]],                              \
        const device uchar* out_mask [[buffer(2)]],                           \
        const device uchar* op_mask [[buffer(3)]],                            \
        device TYPE* output [[buffer(4)]],                                    \
        constant uint& k [[buffer(5)]],                                       \
        constant uint& n [[buffer(6)]],                                       \
        constant uint& ldb [[buffer(7)]],                                     \
        constant uint& lda [[buffer(8)]],                                     \
        constant uint& ldd [[buffer(9)]],                                     \
        uint3 tgid [[threadgroup_position_in_grid]],                          \
        uint simd_gid [[simdgroup_index_in_threadgroup]],                     \
        uint simd_lid [[thread_index_in_simdgroup]]) {                        \
        gemv_masked_impl<TYPE, ACC, ROWS>(                                    \
            matrix, vector, out_mask, op_mask, output,                        \
            k, n, ldb, lda, ldd, tgid, simd_gid, simd_lid);                   \
    }

#define DECL_GEMV_GATHER(NAME, TYPE, ACC, ROWS)                               \
    [[kernel, max_total_threads_per_threadgroup(ROWS * 32)]] void NAME(       \
        const device TYPE* matrix [[buffer(0)]],                              \
        const device TYPE* vector [[buffer(1)]],                              \
        const device uint* vec_indices [[buffer(2)]],                         \
        const device uint* mat_indices [[buffer(3)]],                         \
        device TYPE* output [[buffer(4)]],                                    \
        constant uint& k [[buffer(5)]],                                       \
        constant uint& n [[buffer(6)]],                                       \
        constant uint& ldb [[buffer(7)]],                                     \
        constant uint& lda [[buffer(8)]],                                     \
        constant uint& ldd [[buffer(9)]],                                     \
        uint3 tgid [[threadgroup_position_in_grid]],                          \
        uint simd_gid [[simdgroup_index_in_threadgroup]],                     \
        uint simd_lid [[thread_index_in_simdgroup]]) {                        \
        gemv_gather_impl<TYPE, ACC, ROWS>(                                    \
            matrix, vector, vec_indices, mat_indices, output,                 \
            k, n, ldb, lda, ldd, tgid, simd_gid, simd_lid);                   \
    }

// Masked instantiations
DECL_GEMV_MASKED(gemv_masked_f16_rows4, half, float, 4)
DECL_GEMV_MASKED(gemv_masked_f16_rows8, half, float, 8)
DECL_GEMV_MASKED(gemv_masked_bf16_rows4, bfloat, float, 4)
DECL_GEMV_MASKED(gemv_masked_bf16_rows8, bfloat, float, 8)

// Gather instantiations
DECL_GEMV_GATHER(gemv_gather_f16_rows4, half, float, 4)
DECL_GEMV_GATHER(gemv_gather_f16_rows8, half, float, 8)
DECL_GEMV_GATHER(gemv_gather_bf16_rows4, bfloat, float, 4)
DECL_GEMV_GATHER(gemv_gather_bf16_rows8, bfloat, float, 8)
// Masked/gather GEMV kernels are stubbed out (not used). Keeping file empty to avoid shader compilation errors.
