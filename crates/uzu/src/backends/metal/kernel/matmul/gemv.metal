// SPDX-License-Identifier: MIT

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

#define UZU_MTL_CONST static constant constexpr const

template <typename T>
struct DefaultAccumulator {
    using type = float;
};

template <typename T>
using AccumulatorType = typename DefaultAccumulator<T>::type;

static METAL_FUNC int64_t linear_offset_for_batch(
    uint batch_index,
    const constant int* batch_shape,
    const constant int64_t* batch_strides,
    int batch_ndim
) {
    if (batch_ndim <= 1) {
        return static_cast<int64_t>(batch_index) * batch_strides[0];
    }

    int64_t offset = 0;
    uint remaining = batch_index;
    for (int dim = batch_ndim - 1; dim >= 0; --dim) {
        int dim_size = batch_shape[dim];
        if (dim_size <= 0) {
            continue;
        }
        uint coord = remaining % static_cast<uint>(dim_size);
        remaining /= static_cast<uint>(dim_size);
        offset += coord * batch_strides[dim];
    }
    return offset;
}

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoAxpby, /* Do out = alpha * out + beta * bias */
    typename AccT = AccumulatorType<T>>
struct GEMVKernel {
    using acc_type = AccT;

    UZU_MTL_CONST int threadsM = BM * SM;
    UZU_MTL_CONST int threadsN = BN * SN;

    UZU_MTL_CONST int blockM = threadsM * TM;
    UZU_MTL_CONST int blockN = threadsN * TN;

    static_assert(SM * SN == 32, "simdgroup can only have 32 threads");
    static_assert(
        SN == 4 || SN == 8 || SN == 16 || SN == 32,
        "gemv block must have a width of 4, 8, 16, or 32"
    );

    UZU_MTL_CONST short tgp_mem_size = BN > 1 ? BN * (blockM + TM) : 0;
    UZU_MTL_CONST bool needs_tgp_reduction = BN > 1;

    template <typename U = T>
    static METAL_FUNC void load_unsafe(
        const device T* src,
        thread U dst[TN],
        const int src_offset = 0
    ) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            dst[tn] = static_cast<U>(src[src_offset + tn]);
        }
    }

    template <typename U = T>
    static METAL_FUNC void load_safe(
        const device T* src,
        thread U dst[TN],
        const int src_offset = 0,
        const int src_size = TN
    ) {
        if (src_offset + TN <= src_size) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                dst[tn] = static_cast<U>(src[src_offset + tn]);
            }
        } else {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                dst[tn] = (src_offset + tn) < src_size
                    ? static_cast<U>(src[src_offset + tn])
                    : U(0);
            }
        }
    }

    static METAL_FUNC void run(
        const device T* mat [[buffer(0)]],
        const device T* in_vec [[buffer(1)]],
        const device T* bias [[buffer(2)]],
        device T* out_vec [[buffer(3)]],
        const constant int& in_vec_size [[buffer(4)]],
        const constant int& out_vec_size [[buffer(5)]],
        const constant int& matrix_ld [[buffer(6)]],
        const constant float& alpha [[buffer(7)]],
        const constant float& beta [[buffer(8)]],
        const constant int& bias_stride [[buffer(14)]],
        threadgroup AccT* tgp_memory [[threadgroup(0)]],
        uint3 tid [[threadgroup_position_in_grid]],
        uint3 lid [[thread_position_in_threadgroup]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint simd_lid [[thread_index_in_simdgroup]]
    ) {
        (void)lid;

        thread AccT result[TM] = {0};
        thread T inter[TN];
        thread AccT v_coeff[TN];

        const int thrM = SN != 32 ? simd_lid / SN : 0;
        const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);

        const int sgN = BN != 1 ? (simd_gid % BN) : 0;

        const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
        const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;

        int bm = (simdM + thrM) * TM;
        int bn = (simdN + thrN) * TN;

        int out_row = tid.x * blockM + bm;

        if (out_row >= out_vec_size) {
            return;
        }

        out_row = (out_row + TM <= out_vec_size)
            ? out_row
            : (out_vec_size - TM);

        mat += out_row * matrix_ld;

        constexpr const uniform<int> loop_stride = make_uniform(blockN);
        const uniform<int> in_size = make_uniform(in_vec_size);
        const uniform<int> n_iter = in_size / loop_stride;
        const uniform<int> last_iter = loop_stride * n_iter;
        const uniform<int> leftover = in_size - last_iter;

        for (int i = 0; i < n_iter; ++i) {
            load_unsafe<AccT>(in_vec, v_coeff, bn);

            int mat_offset = 0;
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                load_unsafe(mat, inter, mat_offset + bn);

                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    result[tm] += inter[tn] * v_coeff[tn];
                }

                mat_offset += matrix_ld;
            }

            bn += blockN;
        }

        if (leftover > 0) {
            load_safe<AccT>(in_vec, v_coeff, bn, in_size);

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                load_safe(&mat[tm * matrix_ld], inter, bn, in_size);

                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    result[tm] += inter[tn] * v_coeff[tn];
                }
            }
        }

        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                result[tm] += simd_shuffle_down(result[tm], sn);
            }
        }

        if (needs_tgp_reduction) {
            threadgroup AccT* tgp_results =
                tgp_memory + sgN * (blockM + TM) + bm;
            if (thrN == 0) {
                #pragma unroll
                for (int tm = 0; tm < TM; tm++) {
                    tgp_results[tm] = result[tm];
                }

                threadgroup_barrier(mem_flags::mem_none);

                if (sgN == 0) {
                    #pragma unroll
                    for (int sgn = 1; sgn < BN; sgn++) {
                        #pragma unroll
                        for (int tm = 0; tm < TM; tm++) {
                            result[tm] +=
                                tgp_results[sgn * (blockM + TM) + tm];
                        }
                    }
                }
            }
        }

        if (simdN == 0 && thrN == 0) {
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                if (kDoAxpby) {
                    out_vec[out_row + tm] =
                        static_cast<T>(alpha) * static_cast<T>(result[tm]) +
                        static_cast<T>(beta)
                        * bias[(out_row + tm) * bias_stride];
                } else {
                    out_vec[out_row + tm] = static_cast<T>(result[tm]);
                }
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
/// Vector matrix multiplication (transposed GEMV)
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoAxpby, /* Do out = alpha * out + beta * bias */
    typename AccT = AccumulatorType<T>>
struct GEMVTKernel {
    using acc_type = AccT;

    UZU_MTL_CONST int threadsM = BM * SM;
    UZU_MTL_CONST int threadsN = BN * SN;

    UZU_MTL_CONST int blockM = threadsM * TM;
    UZU_MTL_CONST int blockN = threadsN * TN;

    static_assert(SM * SN == 32, "simdgroup can only have 32 threads");

    UZU_MTL_CONST short tgp_mem_size = BM > 1 ? BM * (blockN + TN) : 0;
    UZU_MTL_CONST bool needs_tgp_reduction = BM > 1;

    static METAL_FUNC void run(
        const device T* mat [[buffer(0)]],
        const device T* in_vec [[buffer(1)]],
        const device T* bias [[buffer(2)]],
        device T* out_vec [[buffer(3)]],
        const constant int& in_vec_size [[buffer(4)]],
        const constant int& out_vec_size [[buffer(5)]],
        const constant int& matrix_ld [[buffer(6)]],
        const constant float& alpha [[buffer(7)]],
        const constant float& beta [[buffer(8)]],
        const constant int& bias_stride [[buffer(14)]],
        threadgroup AccT* tgp_memory [[threadgroup(0)]],
        uint3 tid [[threadgroup_position_in_grid]],
        uint3 lid [[thread_position_in_threadgroup]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint simd_lid [[thread_index_in_simdgroup]]
    ) {
        (void)lid;

        AccT result[TN] = {0};
        T inter[TN];
        AccT v_coeff[TM];
        const int thrM = SN != 32 ? simd_lid / SN : 0;
        const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);

        const int sgM = BN != 1 ? (simd_gid / BN) : int(simd_gid);
        const int sgN = BN != 1 ? (simd_gid % BN) : 0;

        const int simdM = SM * sgM;
        const int simdN = SN * sgN;

        int cm = (simdM + thrM);
        int cn = (simdN + thrN);

        int bm = cm * TM;
        int bn = cn * TN;

        int out_col = tid.x * blockN + bn;

        constexpr const uniform<int> loop_stride = make_uniform(blockM);
        const uniform<int> in_size = make_uniform(in_vec_size);
        const uniform<int> n_iter = in_size / loop_stride;
        const uniform<int> last_iter = loop_stride * n_iter;
        const uniform<int> leftover = in_size - last_iter;

        if (out_col < out_vec_size) {
            out_col = (out_col + TN < out_vec_size)
                ? out_col
                : (out_vec_size - TN);

            for (int i = 0; i < n_iter; ++i) {
                threadgroup_barrier(mem_flags::mem_none);

                #pragma unroll
                for (int tm = 0; tm < TM; tm++) {
                    v_coeff[tm] = static_cast<AccT>(in_vec[bm + tm]);
                }

                #pragma unroll
                for (int tm = 0; tm < TM; tm++) {
                    auto vc = static_cast<AccT>(v_coeff[tm]);
                    for (int tn = 0; tn < TN; tn++) {
                        inter[tn] = mat[(bm + tm) * matrix_ld + out_col + tn];
                    }
                    for (int tn = 0; tn < TN; tn++) {
                        result[tn] += vc * inter[tn];
                    }
                }

                bm += blockM;
            }

            if (leftover > 0) {
                for (int tm = 0; tm < TM && bm + tm < in_vec_size; tm++) {
                    v_coeff[tm] = static_cast<AccT>(in_vec[bm + tm]);

                    #pragma unroll
                    for (int tn = 0; tn < TN; tn++) {
                        inter[tn] =
                            mat[(bm + tm) * matrix_ld + out_col + tn];
                    }

                    #pragma unroll
                    for (int tn = 0; tn < TN; tn++) {
                        result[tn] += v_coeff[tm] * inter[tn];
                    }
                }
            }
        }

        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            #pragma unroll
            for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) {
                result[tn] += simd_shuffle_down(result[tn], SN * sm);
            }
        }

        if (needs_tgp_reduction) {
            threadgroup AccT* tgp_results =
                tgp_memory + sgM * (blockN + TN) + bn;
            if (thrM == 0) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    tgp_results[tn] = result[tn];
                }

                threadgroup_barrier(mem_flags::mem_none);

                if (sgM == 0) {
                    #pragma unroll
                    for (int sgm = 1; sgm < BM; sgm++) {
                        #pragma unroll
                        for (int tn = 0; tn < TN; tn++) {
                            result[tn] +=
                                tgp_results[sgm * (blockN + TN) + tn];
                        }
                    }
                }
            }
        }

        if (cm == 0 && out_col < out_vec_size) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                if (kDoAxpby) {
                    out_vec[out_col + j] =
                        static_cast<T>(alpha) * static_cast<T>(result[j]) +
                        static_cast<T>(beta)
                        * bias[(out_col + j) * bias_stride];
                } else {
                    out_vec[out_col + j] = static_cast<T>(result[j]);
                }
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
/// Kernel entry points
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoNCBatch, /* Batch ndim > 1 */
    const bool kDoAxpby> /* Do out = alpha * out + beta * bias */
METAL_FUNC void gemv_entry(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    threadgroup typename GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>::acc_type* tgp_memory,
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    using gemv_kernel = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;

    if (kDoNCBatch) {
        in_vec += linear_offset_for_batch(
            tid.z, batch_shape, vector_batch_stride, batch_ndim
        );
        mat += linear_offset_for_batch(
            tid.z, batch_shape, matrix_batch_stride, batch_ndim
        );
        if (kDoAxpby) {
            bias += linear_offset_for_batch(
                tid.z, batch_shape, bias_batch_stride, batch_ndim
            );
        }
    } else {
        in_vec += tid.z * vector_batch_stride[0];
        mat += tid.z * matrix_batch_stride[0];
        if (kDoAxpby) {
            bias += tid.z * bias_batch_stride[0];
        }
    }

    out_vec += tid.z * out_vec_size;

    gemv_kernel::run(
        mat,
        in_vec,
        bias,
        out_vec,
        in_vec_size,
        out_vec_size,
        matrix_ld,
        alpha,
        beta,
        bias_stride,
        tgp_memory,
        tid,
        lid,
        simd_gid,
        simd_lid
    );
}

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoNCBatch, /* Batch ndim > 1 */
    const bool kDoAxpby> /* Do out = alpha * out + beta * bias */
METAL_FUNC void gemv_t_entry(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    threadgroup typename GEMVTKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>::acc_type* tgp_memory,
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    using gemv_kernel = GEMVTKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;

    if (kDoNCBatch) {
        in_vec += linear_offset_for_batch(
            tid.z, batch_shape, vector_batch_stride, batch_ndim
        );
        mat += linear_offset_for_batch(
            tid.z, batch_shape, matrix_batch_stride, batch_ndim
        );
        if (kDoAxpby) {
            bias += linear_offset_for_batch(
                tid.z, batch_shape, bias_batch_stride, batch_ndim
            );
        }
    } else {
        in_vec += tid.z * vector_batch_stride[0];
        mat += tid.z * matrix_batch_stride[0];
        if (kDoAxpby) {
            bias += tid.z * bias_batch_stride[0];
        }
    }

    out_vec += tid.z * out_vec_size;

    gemv_kernel::run(
        mat,
        in_vec,
        bias,
        out_vec,
        in_vec_size,
        out_vec_size,
        matrix_ld,
        alpha,
        beta,
        bias_stride,
        tgp_memory,
        tid,
        lid,
        simd_gid,
        simd_lid);
}

///////////////////////////////////////////////////////////////////////////////
/// Instantiation helpers
///////////////////////////////////////////////////////////////////////////////

#define instantiate_gemv_helper(                                      \
    name, itype, bm, bn, sm, sn, tm, tn, nc, axpby)                   \
  [[host_name(                                                        \
      "gemv_" #name "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn "_tm" #tm \
      "_tn" #tn "_nc" #nc "_axpby" #axpby)]]                          \
  [[kernel, max_total_threads_per_threadgroup(bm * bn * 32)]]         \
  void gemv_inst_##name##_bm##bm##_bn##bn##_sm##sm##_sn##sn##_tm##tm##_tn##tn##_nc##nc##_axpby##axpby( \
      const device itype* mat [[buffer(0)]],                          \
      const device itype* in_vec [[buffer(1)]],                       \
      const device itype* bias [[buffer(2)]],                         \
      device itype* out_vec [[buffer(3)]],                            \
      const constant int& in_vec_size [[buffer(4)]],                  \
      const constant int& out_vec_size [[buffer(5)]],                 \
      const constant int& matrix_ld [[buffer(6)]],                    \
      const constant float& alpha [[buffer(7)]],                      \
      const constant float& beta [[buffer(8)]],                       \
      const constant int& batch_ndim [[buffer(9)]],                   \
      const constant int* batch_shape [[buffer(10)]],                 \
      const constant int64_t* vector_batch_stride [[buffer(11)]],     \
      const constant int64_t* matrix_batch_stride [[buffer(12)]],     \
      const constant int64_t* bias_batch_stride [[buffer(13)]],       \
      const constant int& bias_stride [[buffer(14)]],                 \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint3 lid [[thread_position_in_threadgroup]],                   \
      uint simd_gid [[simdgroup_index_in_threadgroup]],               \
      uint simd_lid [[thread_index_in_simdgroup]]) {                  \
    using gemv_kernel = GEMVKernel<itype, bm, bn, sm, sn, tm, tn, axpby>; \
    threadgroup typename gemv_kernel::acc_type tgp_memory[            \
        gemv_kernel::tgp_mem_size == 0 ? 1 : gemv_kernel::tgp_mem_size]; \
    gemv_entry<itype, bm, bn, sm, sn, tm, tn, nc, axpby>(             \
        mat,                                                          \
        in_vec,                                                       \
        bias,                                                         \
        out_vec,                                                      \
        in_vec_size,                                                  \
        out_vec_size,                                                 \
        matrix_ld,                                                    \
        alpha,                                                        \
        beta,                                                         \
        batch_ndim,                                                   \
        batch_shape,                                                  \
        vector_batch_stride,                                          \
        matrix_batch_stride,                                          \
        bias_batch_stride,                                            \
        bias_stride,                                                  \
        gemv_kernel::tgp_mem_size == 0 ? nullptr : tgp_memory,        \
        tid,                                                          \
        lid,                                                          \
        simd_gid,                                                     \
        simd_lid);                                                    \
  }                                                                   \
  [[host_name(                                                        \
      "gemv_t_" #name "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn "_tm"   \
      #tm "_tn" #tn "_nc" #nc "_axpby" #axpby)]]                      \
  [[kernel, max_total_threads_per_threadgroup(bm * bn * 32)]]         \
  void gemv_t_inst_##name##_bm##bm##_bn##bn##_sm##sm##_sn##sn##_tm##tm##_tn##tn##_nc##nc##_axpby##axpby( \
      const device itype* mat [[buffer(0)]],                          \
      const device itype* in_vec [[buffer(1)]],                       \
      const device itype* bias [[buffer(2)]],                         \
      device itype* out_vec [[buffer(3)]],                            \
      const constant int& in_vec_size [[buffer(4)]],                  \
      const constant int& out_vec_size [[buffer(5)]],                 \
      const constant int& matrix_ld [[buffer(6)]],                    \
      const constant float& alpha [[buffer(7)]],                      \
      const constant float& beta [[buffer(8)]],                       \
      const constant int& batch_ndim [[buffer(9)]],                   \
      const constant int* batch_shape [[buffer(10)]],                 \
      const constant int64_t* vector_batch_stride [[buffer(11)]],     \
      const constant int64_t* matrix_batch_stride [[buffer(12)]],     \
      const constant int64_t* bias_batch_stride [[buffer(13)]],       \
      const constant int& bias_stride [[buffer(14)]],                 \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint3 lid [[thread_position_in_threadgroup]],                   \
      uint simd_gid [[simdgroup_index_in_threadgroup]],               \
      uint simd_lid [[thread_index_in_simdgroup]]) {                  \
    using gemv_kernel = GEMVTKernel<itype, bm, bn, sm, sn, tm, tn, axpby>; \
    threadgroup typename gemv_kernel::acc_type tgp_memory[            \
        gemv_kernel::tgp_mem_size == 0 ? 1 : gemv_kernel::tgp_mem_size]; \
    gemv_t_entry<itype, bm, bn, sm, sn, tm, tn, nc, axpby>(           \
        mat,                                                          \
        in_vec,                                                       \
        bias,                                                         \
        out_vec,                                                      \
        in_vec_size,                                                  \
        out_vec_size,                                                 \
        matrix_ld,                                                    \
        alpha,                                                        \
        beta,                                                         \
        batch_ndim,                                                   \
        batch_shape,                                                  \
        vector_batch_stride,                                          \
        matrix_batch_stride,                                          \
        bias_batch_stride,                                            \
        bias_stride,                                                  \
        gemv_kernel::tgp_mem_size == 0 ? nullptr : tgp_memory,        \
        tid,                                                          \
        lid,                                                          \
        simd_gid,                                                     \
        simd_lid);                                                    \
  }

// clang-format off
#define instantiate_gemv(name, itype, bm, bn, sm, sn, tm, tn)        \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 0, 0) \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 0, 1) \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 1, 0) \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 1, 1)

#define instantiate_gemv_blocks(name, itype) \
  instantiate_gemv(name, itype, 1,  8, 1, 32, 4, 4) \
  instantiate_gemv(name, itype, 1,  8, 1, 32, 1, 4) \
  instantiate_gemv(name, itype, 1,  1, 8,  4, 4, 4) \
  instantiate_gemv(name, itype, 1,  1, 8,  4, 1, 4) \
  instantiate_gemv(name, itype, 4,  1, 1, 32, 1, 4) \
  instantiate_gemv(name, itype, 4,  1, 1, 32, 4, 4) \
  instantiate_gemv(name, itype, 8,  1, 1, 32, 4, 4)

instantiate_gemv_blocks(float32, float);
instantiate_gemv_blocks(float16, half);
instantiate_gemv_blocks(bfloat16, bfloat);

#define instantiate_gemv_t(name, itype, bm, bn, sm, sn, tm, tn)        \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 0, 0)   \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 0, 1)   \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 1, 0)   \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 1, 1)

#define instantiate_gemv_t_blocks(name, itype) \
  instantiate_gemv_t(name, itype, 1, 2,  8, 4, 4, 1) \
  instantiate_gemv_t(name, itype, 1, 2,  8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 4,  8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 16, 8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 16, 4, 8, 4, 4)

instantiate_gemv_t_blocks(float32, float);
instantiate_gemv_t_blocks(float16, half);
instantiate_gemv_t_blocks(bfloat16, bfloat);
// clang-format on
