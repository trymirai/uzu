#include <metal_simdgroup>
#include <metal_stdlib>

#include "../../../common/utils.h"
#include "../../../definitions.metal"

#include "../../common/steel/utils.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication
///////////////////////////////////////////////////////////////////////////////

#define MTL_CONST static constant constexpr const

template <typename U>
struct DefaultAccT {
  using type = float;
};
template <>
struct DefaultAccT<complex64_t> {
  using type = complex64_t;
};

template <
    typename T,
    const int BM,        /* Threadgroup rows (in simdgroups) */
    const int BN,        /* Threadgroup cols (in simdgroups) */
    const int SM,        /* Simdgroup rows (in threads) */
    const int SN,        /* Simdgroup cols (in threads) */
    const int TM,        /* Thread rows (in elements) */
    const int TN,        /* Thread cols (in elements) */
    const bool kDoAxpby, /* Do out = alpha * out + beta * bias */
    const int BP,        /* Batch rows per threadgroup */
    typename AccT = typename DefaultAccT<T>::type>
struct GEMVKernel {
  using acc_type = AccT;

  MTL_CONST int threadsM = BM * SM;
  MTL_CONST int threadsN = BN * SN;

  MTL_CONST int blockM = threadsM * TM;
  MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup can only have 32 threads");

  static_assert(
      SN == 4 || SN == 8 || SN == 16 || SN == 32,
      "gemv block must have a width of 4, 8, 16, or 32"
  );

  // - The matrix of size (M = out_vec_size, K = in_vec_size) is divided up
  //   into blocks of (blockM, blockN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each threadgroup has (threadsN, threadsM, 1) threads
  //
  // 1. A thread loads TN elements each from mat along TM rows
  //    and the corresponding scalar from the vector
  // 2. The thread then multiplies and adds to accumulate its local result for
  //    the block
  // 3. At the end, each thread has accumulated results over all blocks across
  //    the rows. These are then summed up across the threadgroup
  // 4. Each threadgroup writes its accumulated blockM outputs
  //
  // Edge case handling:
  // - The threadgroup with the largest tid has blocks that exceed the matrix
  //   * The blocks that start outside the matrix are never read (thread results
  //     remain zero)
  //   * The last thread that partially overlaps with the matrix is shifted
  //     inwards such that the thread block fits exactly in the matrix

  MTL_CONST short tgp_mem_size = BN > 1 ? BN*(blockM + TM) * BP : 0;
  MTL_CONST bool needs_tgp_reduction = BN > 1;

  template <typename U = T>
  static METAL_FUNC void load_unsafe(
      const device T* src,
      thread U dst[TN],
      const int src_offset = 0
  ) {
    MTL_PRAGMA_UNROLL
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
      MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = static_cast<U>(src[src_offset + tn]);
      }
    } else { // Edgecase
      MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = src_offset + tn < src_size
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
      const int batch_row_base,
      threadgroup AccT* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]
  ) {
    // Appease compiler
    (void)lid;

    // Thread local accumulation results
    thread AccT result[BP][TM] = {{0}};
    thread T inter[TN];
    thread AccT v_coeff[BP][TN];

    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);

    const int sgN = BN != 1 ? (simd_gid % BN) : 0;

    const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
    const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;

    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;

    // Block position
    int out_row = tid.x * blockM + bm;

    // Exit simdgroup if rows out of bound
    if (out_row >= out_vec_size)
      return;

    // Adjust tail simdgroup to ensure in bound reads
    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;

    // Advance matrix
    mat += out_row * matrix_ld;

    constexpr const uniform<int> loop_stride = make_uniform(blockN);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    const uniform<int> last_iter = loop_stride * n_iter;
    const uniform<int> leftover = in_size - last_iter;

    // Loop over in_vec in blocks of blockN
    for (int i = 0; i < n_iter; ++i) {
      MTL_PRAGMA_UNROLL
      for (int bp = 0; bp < BP; bp++) {
        const int batch_row = batch_row_base + bp;
        const device T* vec_ptr = in_vec + batch_row * in_vec_size;
        load_unsafe<AccT>(vec_ptr, v_coeff[bp], bn);
      }

      // Per thread work loop
      int mat_offset = 0;
      MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        // Load for the row
        load_unsafe(mat, inter, mat_offset + bn);

        // Accumulate results
        MTL_PRAGMA_UNROLL
        for (int bp = 0; bp < BP; bp++) {
          MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            result[bp][tm] += inter[tn] * v_coeff[bp][tn];
          }
        }

        mat_offset += matrix_ld;
      }

      bn += blockN;
    }

    if (leftover > 0) {
      MTL_PRAGMA_UNROLL
      for (int bp = 0; bp < BP; bp++) {
        const int batch_row = batch_row_base + bp;
        const device T* vec_ptr = in_vec + batch_row * in_vec_size;
        load_safe<AccT>(vec_ptr, v_coeff[bp], bn, in_size);
      }

      // Per thread work loop
      MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        // Load for the row
        load_safe(&mat[tm * matrix_ld], inter, bn, in_size);

        // Accumulate results
        MTL_PRAGMA_UNROLL
        for (int bp = 0; bp < BP; bp++) {
          MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            result[bp][tm] += inter[tn] * v_coeff[bp][tn];
          }
        }
      }
    }

    // Simdgroup accumulations
    MTL_PRAGMA_UNROLL
    for (int bp = 0; bp < BP; bp++) {
      MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        MTL_PRAGMA_UNROLL
        for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
          result[bp][tm] += simd_shuffle_down(result[bp][tm], sn);
        }
      }
    }

    // Threadgroup accumulation results
    if (needs_tgp_reduction) {
      if (thrN == 0) {
        MTL_PRAGMA_UNROLL
        for (int bp = 0; bp < BP; bp++) {
          threadgroup AccT* tgp_results =
              tgp_memory + bp * (BN * (blockM + TM)) + sgN * (blockM + TM) + bm;
          MTL_PRAGMA_UNROLL
          for (int tm = 0; tm < TM; tm++) {
            tgp_results[tm] = result[bp][tm];
          }
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (sgN == 0) {
          MTL_PRAGMA_UNROLL
          for (int bp = 0; bp < BP; bp++) {
            threadgroup AccT* tgp_results =
                tgp_memory + bp * (BN * (blockM + TM)) + bm;
            MTL_PRAGMA_UNROLL
            for (int sgn = 1; sgn < BN; sgn++) {
              MTL_PRAGMA_UNROLL
              for (int tm = 0; tm < TM; tm++) {
                result[bp][tm] += tgp_results[sgn * (blockM + TM) + tm];
              }
            }
          }
        }
      }
    }

    // Write outputs
    if (simdN == 0 && thrN == 0) {
      MTL_PRAGMA_UNROLL
      for (int bp = 0; bp < BP; bp++) {
        const int batch_row = batch_row_base + bp;
        device T* out_row_ptr = out_vec + batch_row * out_vec_size;
        MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          if (kDoAxpby) {
            out_row_ptr[out_row + tm] =
                static_cast<T>(alpha) * static_cast<T>(result[bp][tm]) +
                static_cast<T>(beta) * bias[(out_row + tm) * bias_stride];
          } else {
            out_row_ptr[out_row + tm] = static_cast<T>(result[bp][tm]);
          }
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// DSL GEMV kernels
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int threadgroup_simd_rows,
    const int threadgroup_simd_cols,
    const int simdgroup_thread_rows,
    const int simdgroup_thread_cols,
    const int thread_output_rows,
    const int thread_output_cols,
    const bool apply_output_scale_and_accumulate>
inline void run_matmul_gemv_shape(
    const device T* matrix,
    const device T* input_vector,
    const device T* output_source,
    device T* output_vector,
    const constant int& input_dimension,
    const constant int& output_dimension,
    const constant int& matrix_leading_dimension,
    const constant float& output_scale,
    const constant float& output_accumulate_scale,
    const constant int* vector_batch_stride,
    const constant int* matrix_batch_stride,
    const constant int* output_source_batch_stride,
    const constant int& output_source_stride,
    const constant int& batch_rows,
    threadgroup typename GEMVKernel<
        T,
        threadgroup_simd_rows,
        threadgroup_simd_cols,
        simdgroup_thread_rows,
        simdgroup_thread_cols,
        thread_output_rows,
        thread_output_cols,
        false,
        1>::acc_type* threadgroup_memory,
    const uint3 threadgroup_position,
    const uint3 thread_position,
    const uint simd_group_index,
    const uint simd_lane_index
) {
  using gemv_kernel = GEMVKernel<
      T,
      threadgroup_simd_rows,
      threadgroup_simd_cols,
      simdgroup_thread_rows,
      simdgroup_thread_cols,
      thread_output_rows,
      thread_output_cols,
      apply_output_scale_and_accumulate,
      1>;

  const int batch_row = static_cast<int>(threadgroup_position.y);
  if (batch_row >= batch_rows) {
    return;
  }

  input_vector += threadgroup_position.z * vector_batch_stride[0];
  matrix += threadgroup_position.z * matrix_batch_stride[0];
  IF_CONSTEXPR(apply_output_scale_and_accumulate) {
    output_source += threadgroup_position.z * output_source_batch_stride[0];
  }

  output_vector += threadgroup_position.z * batch_rows * output_dimension;

  gemv_kernel::run(
      matrix,
      input_vector,
      output_source,
      output_vector,
      input_dimension,
      output_dimension,
      matrix_leading_dimension,
      output_scale,
      output_accumulate_scale,
      output_source_stride,
      batch_row,
      gemv_kernel::tgp_mem_size == 0 ? nullptr : threadgroup_memory,
      threadgroup_position,
      thread_position,
      simd_group_index,
      simd_lane_index
  );
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemvShape0)(
    const device T* matrix,
    const device T* input_vector,
    const device T* output_source OPTIONAL(apply_output_scale_and_accumulate),
    device T* output_vector,
    const constant int& input_dimension,
    const constant int& output_dimension,
    const constant int& matrix_leading_dimension,
    const constant float& output_scale,
    const constant float& output_accumulate_scale,
    const constant int& batch_ndim,
    const constant int* batch_shape,
    const constant int* vector_batch_stride,
    const constant int* matrix_batch_stride,
    const constant int* output_source_batch_stride,
    const constant int& output_source_stride,
    const constant int& batch_rows,
    const constant int& output_leading_dimension,
    const constant int& vector_leading_dimension,
    threadgroup typename GEMVKernel<T, 1, 8, 1, 32, 4, 4, false, 1>::acc_type
        threadgroup_memory[GEMVKernel<T, 1, 8, 1, 32, 4, 4, false, 1>::tgp_mem_size == 0
            ? 1
            : GEMVKernel<T, 1, 8, 1, 32, 4, 4, false, 1>::tgp_mem_size],
    const bool apply_output_scale_and_accumulate SPECIALIZE,
    const uint threadgroup_index_x GROUPS((output_dimension + 4 - 1) / 4),
    const uint threadgroup_index_y GROUPS(batch_rows),
    const uint threadgroup_index_z GROUPS(batch_shape[0]),
    const uint thread_index_x THREADS(32),
    const uint thread_index_y THREADS(8),
    const uint thread_index_z THREADS(1),
    const Simd simd
) {
  if (batch_ndim == 0 || output_leading_dimension == 0 || vector_leading_dimension == 0) {
    return;
  }
  const uint3 threadgroup_position = uint3(threadgroup_index_x, threadgroup_index_y, threadgroup_index_z);
  const uint3 thread_position = uint3(thread_index_x, thread_index_y, thread_index_z);
  if (apply_output_scale_and_accumulate) {
    run_matmul_gemv_shape<T, 1, 8, 1, 32, 4, 4, true>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension,
        output_dimension,
        matrix_leading_dimension,
        output_scale,
        output_accumulate_scale,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride,
        batch_rows,
        threadgroup_memory,
        threadgroup_position,
        thread_position,
        simd.group_idx,
        simd.lane_idx
    );
  } else {
    run_matmul_gemv_shape<T, 1, 8, 1, 32, 4, 4, false>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension,
        output_dimension,
        matrix_leading_dimension,
        output_scale,
        output_accumulate_scale,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride,
        batch_rows,
        threadgroup_memory,
        threadgroup_position,
        thread_position,
        simd.group_idx,
        simd.lane_idx
    );
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemvShape1)(
    const device T* matrix,
    const device T* input_vector,
    const device T* output_source OPTIONAL(apply_output_scale_and_accumulate),
    device T* output_vector,
    const constant int& input_dimension,
    const constant int& output_dimension,
    const constant int& matrix_leading_dimension,
    const constant float& output_scale,
    const constant float& output_accumulate_scale,
    const constant int& batch_ndim,
    const constant int* batch_shape,
    const constant int* vector_batch_stride,
    const constant int* matrix_batch_stride,
    const constant int* output_source_batch_stride,
    const constant int& output_source_stride,
    const constant int& batch_rows,
    const constant int& output_leading_dimension,
    const constant int& vector_leading_dimension,
    threadgroup typename GEMVKernel<T, 1, 8, 1, 32, 1, 4, false, 1>::acc_type
        threadgroup_memory[GEMVKernel<T, 1, 8, 1, 32, 1, 4, false, 1>::tgp_mem_size == 0
            ? 1
            : GEMVKernel<T, 1, 8, 1, 32, 1, 4, false, 1>::tgp_mem_size],
    const bool apply_output_scale_and_accumulate SPECIALIZE,
    const uint threadgroup_index_x GROUPS((output_dimension + 1 - 1) / 1),
    const uint threadgroup_index_y GROUPS(batch_rows),
    const uint threadgroup_index_z GROUPS(batch_shape[0]),
    const uint thread_index_x THREADS(32),
    const uint thread_index_y THREADS(8),
    const uint thread_index_z THREADS(1),
    const Simd simd
) {
  if (batch_ndim == 0 || output_leading_dimension == 0 || vector_leading_dimension == 0) {
    return;
  }
  const uint3 threadgroup_position = uint3(threadgroup_index_x, threadgroup_index_y, threadgroup_index_z);
  const uint3 thread_position = uint3(thread_index_x, thread_index_y, thread_index_z);
  if (apply_output_scale_and_accumulate) {
    run_matmul_gemv_shape<T, 1, 8, 1, 32, 1, 4, true>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension,
        output_dimension,
        matrix_leading_dimension,
        output_scale,
        output_accumulate_scale,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride,
        batch_rows,
        threadgroup_memory,
        threadgroup_position,
        thread_position,
        simd.group_idx,
        simd.lane_idx
    );
  } else {
    run_matmul_gemv_shape<T, 1, 8, 1, 32, 1, 4, false>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension,
        output_dimension,
        matrix_leading_dimension,
        output_scale,
        output_accumulate_scale,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride,
        batch_rows,
        threadgroup_memory,
        threadgroup_position,
        thread_position,
        simd.group_idx,
        simd.lane_idx
    );
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemvShape2)(
    const device T* matrix,
    const device T* input_vector,
    const device T* output_source OPTIONAL(apply_output_scale_and_accumulate),
    device T* output_vector,
    const constant int& input_dimension,
    const constant int& output_dimension,
    const constant int& matrix_leading_dimension,
    const constant float& output_scale,
    const constant float& output_accumulate_scale,
    const constant int& batch_ndim,
    const constant int* batch_shape,
    const constant int* vector_batch_stride,
    const constant int* matrix_batch_stride,
    const constant int* output_source_batch_stride,
    const constant int& output_source_stride,
    const constant int& batch_rows,
    const constant int& output_leading_dimension,
    const constant int& vector_leading_dimension,
    threadgroup typename GEMVKernel<T, 1, 1, 8, 4, 4, 4, false, 1>::acc_type
        threadgroup_memory[GEMVKernel<T, 1, 1, 8, 4, 4, 4, false, 1>::tgp_mem_size == 0
            ? 1
            : GEMVKernel<T, 1, 1, 8, 4, 4, 4, false, 1>::tgp_mem_size],
    const bool apply_output_scale_and_accumulate SPECIALIZE,
    const uint threadgroup_index_x GROUPS((output_dimension + 32 - 1) / 32),
    const uint threadgroup_index_y GROUPS(batch_rows),
    const uint threadgroup_index_z GROUPS(batch_shape[0]),
    const uint thread_index_x THREADS(32),
    const uint thread_index_y THREADS(1),
    const uint thread_index_z THREADS(1),
    const Simd simd
) {
  if (batch_ndim == 0 || output_leading_dimension == 0 || vector_leading_dimension == 0) {
    return;
  }
  const uint3 threadgroup_position = uint3(threadgroup_index_x, threadgroup_index_y, threadgroup_index_z);
  const uint3 thread_position = uint3(thread_index_x, thread_index_y, thread_index_z);
  if (apply_output_scale_and_accumulate) {
    run_matmul_gemv_shape<T, 1, 1, 8, 4, 4, 4, true>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension,
        output_dimension,
        matrix_leading_dimension,
        output_scale,
        output_accumulate_scale,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride,
        batch_rows,
        threadgroup_memory,
        threadgroup_position,
        thread_position,
        simd.group_idx,
        simd.lane_idx
    );
  } else {
    run_matmul_gemv_shape<T, 1, 1, 8, 4, 4, 4, false>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension,
        output_dimension,
        matrix_leading_dimension,
        output_scale,
        output_accumulate_scale,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride,
        batch_rows,
        threadgroup_memory,
        threadgroup_position,
        thread_position,
        simd.group_idx,
        simd.lane_idx
    );
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemvShape3)(
    const device T* matrix,
    const device T* input_vector,
    const device T* output_source OPTIONAL(apply_output_scale_and_accumulate),
    device T* output_vector,
    const constant int& input_dimension,
    const constant int& output_dimension,
    const constant int& matrix_leading_dimension,
    const constant float& output_scale,
    const constant float& output_accumulate_scale,
    const constant int& batch_ndim,
    const constant int* batch_shape,
    const constant int* vector_batch_stride,
    const constant int* matrix_batch_stride,
    const constant int* output_source_batch_stride,
    const constant int& output_source_stride,
    const constant int& batch_rows,
    const constant int& output_leading_dimension,
    const constant int& vector_leading_dimension,
    threadgroup typename GEMVKernel<T, 1, 1, 8, 4, 1, 4, false, 1>::acc_type
        threadgroup_memory[GEMVKernel<T, 1, 1, 8, 4, 1, 4, false, 1>::tgp_mem_size == 0
            ? 1
            : GEMVKernel<T, 1, 1, 8, 4, 1, 4, false, 1>::tgp_mem_size],
    const bool apply_output_scale_and_accumulate SPECIALIZE,
    const uint threadgroup_index_x GROUPS((output_dimension + 8 - 1) / 8),
    const uint threadgroup_index_y GROUPS(batch_rows),
    const uint threadgroup_index_z GROUPS(batch_shape[0]),
    const uint thread_index_x THREADS(32),
    const uint thread_index_y THREADS(1),
    const uint thread_index_z THREADS(1),
    const Simd simd
) {
  if (batch_ndim == 0 || output_leading_dimension == 0 || vector_leading_dimension == 0) {
    return;
  }
  const uint3 threadgroup_position = uint3(threadgroup_index_x, threadgroup_index_y, threadgroup_index_z);
  const uint3 thread_position = uint3(thread_index_x, thread_index_y, thread_index_z);
  if (apply_output_scale_and_accumulate) {
    run_matmul_gemv_shape<T, 1, 1, 8, 4, 1, 4, true>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension,
        output_dimension,
        matrix_leading_dimension,
        output_scale,
        output_accumulate_scale,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride,
        batch_rows,
        threadgroup_memory,
        threadgroup_position,
        thread_position,
        simd.group_idx,
        simd.lane_idx
    );
  } else {
    run_matmul_gemv_shape<T, 1, 1, 8, 4, 1, 4, false>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension,
        output_dimension,
        matrix_leading_dimension,
        output_scale,
        output_accumulate_scale,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride,
        batch_rows,
        threadgroup_memory,
        threadgroup_position,
        thread_position,
        simd.group_idx,
        simd.lane_idx
    );
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemvShape4)(
    const device T* matrix,
    const device T* input_vector,
    const device T* output_source OPTIONAL(apply_output_scale_and_accumulate),
    device T* output_vector,
    const constant int& input_dimension,
    const constant int& output_dimension,
    const constant int& matrix_leading_dimension,
    const constant float& output_scale,
    const constant float& output_accumulate_scale,
    const constant int& batch_ndim,
    const constant int* batch_shape,
    const constant int* vector_batch_stride,
    const constant int* matrix_batch_stride,
    const constant int* output_source_batch_stride,
    const constant int& output_source_stride,
    const constant int& batch_rows,
    const constant int& output_leading_dimension,
    const constant int& vector_leading_dimension,
    threadgroup typename GEMVKernel<T, 4, 1, 1, 32, 1, 4, false, 1>::acc_type
        threadgroup_memory[GEMVKernel<T, 4, 1, 1, 32, 1, 4, false, 1>::tgp_mem_size == 0
            ? 1
            : GEMVKernel<T, 4, 1, 1, 32, 1, 4, false, 1>::tgp_mem_size],
    const bool apply_output_scale_and_accumulate SPECIALIZE,
    const uint threadgroup_index_x GROUPS((output_dimension + 4 - 1) / 4),
    const uint threadgroup_index_y GROUPS(batch_rows),
    const uint threadgroup_index_z GROUPS(batch_shape[0]),
    const uint thread_index_x THREADS(32),
    const uint thread_index_y THREADS(1),
    const uint thread_index_z THREADS(4),
    const Simd simd
) {
  if (batch_ndim == 0 || output_leading_dimension == 0 || vector_leading_dimension == 0) {
    return;
  }
  const uint3 threadgroup_position = uint3(threadgroup_index_x, threadgroup_index_y, threadgroup_index_z);
  const uint3 thread_position = uint3(thread_index_x, thread_index_y, thread_index_z);
  if (apply_output_scale_and_accumulate) {
    run_matmul_gemv_shape<T, 4, 1, 1, 32, 1, 4, true>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension,
        output_dimension,
        matrix_leading_dimension,
        output_scale,
        output_accumulate_scale,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride,
        batch_rows,
        threadgroup_memory,
        threadgroup_position,
        thread_position,
        simd.group_idx,
        simd.lane_idx
    );
  } else {
    run_matmul_gemv_shape<T, 4, 1, 1, 32, 1, 4, false>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension,
        output_dimension,
        matrix_leading_dimension,
        output_scale,
        output_accumulate_scale,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride,
        batch_rows,
        threadgroup_memory,
        threadgroup_position,
        thread_position,
        simd.group_idx,
        simd.lane_idx
    );
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemvShape5)(
    const device T* matrix,
    const device T* input_vector,
    const device T* output_source OPTIONAL(apply_output_scale_and_accumulate),
    device T* output_vector,
    const constant int& input_dimension,
    const constant int& output_dimension,
    const constant int& matrix_leading_dimension,
    const constant float& output_scale,
    const constant float& output_accumulate_scale,
    const constant int& batch_ndim,
    const constant int* batch_shape,
    const constant int* vector_batch_stride,
    const constant int* matrix_batch_stride,
    const constant int* output_source_batch_stride,
    const constant int& output_source_stride,
    const constant int& batch_rows,
    const constant int& output_leading_dimension,
    const constant int& vector_leading_dimension,
    threadgroup typename GEMVKernel<T, 4, 1, 1, 32, 4, 4, false, 1>::acc_type
        threadgroup_memory[GEMVKernel<T, 4, 1, 1, 32, 4, 4, false, 1>::tgp_mem_size == 0
            ? 1
            : GEMVKernel<T, 4, 1, 1, 32, 4, 4, false, 1>::tgp_mem_size],
    const bool apply_output_scale_and_accumulate SPECIALIZE,
    const uint threadgroup_index_x GROUPS((output_dimension + 16 - 1) / 16),
    const uint threadgroup_index_y GROUPS(batch_rows),
    const uint threadgroup_index_z GROUPS(batch_shape[0]),
    const uint thread_index_x THREADS(32),
    const uint thread_index_y THREADS(1),
    const uint thread_index_z THREADS(4),
    const Simd simd
) {
  if (batch_ndim == 0 || output_leading_dimension == 0 || vector_leading_dimension == 0) {
    return;
  }
  const uint3 threadgroup_position = uint3(threadgroup_index_x, threadgroup_index_y, threadgroup_index_z);
  const uint3 thread_position = uint3(thread_index_x, thread_index_y, thread_index_z);
  if (apply_output_scale_and_accumulate) {
    run_matmul_gemv_shape<T, 4, 1, 1, 32, 4, 4, true>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension,
        output_dimension,
        matrix_leading_dimension,
        output_scale,
        output_accumulate_scale,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride,
        batch_rows,
        threadgroup_memory,
        threadgroup_position,
        thread_position,
        simd.group_idx,
        simd.lane_idx
    );
  } else {
    run_matmul_gemv_shape<T, 4, 1, 1, 32, 4, 4, false>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension,
        output_dimension,
        matrix_leading_dimension,
        output_scale,
        output_accumulate_scale,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride,
        batch_rows,
        threadgroup_memory,
        threadgroup_position,
        thread_position,
        simd.group_idx,
        simd.lane_idx
    );
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemvShape6)(
    const device T* matrix,
    const device T* input_vector,
    const device T* output_source OPTIONAL(apply_output_scale_and_accumulate),
    device T* output_vector,
    const constant int& input_dimension,
    const constant int& output_dimension,
    const constant int& matrix_leading_dimension,
    const constant float& output_scale,
    const constant float& output_accumulate_scale,
    const constant int& batch_ndim,
    const constant int* batch_shape,
    const constant int* vector_batch_stride,
    const constant int* matrix_batch_stride,
    const constant int* output_source_batch_stride,
    const constant int& output_source_stride,
    const constant int& batch_rows,
    const constant int& output_leading_dimension,
    const constant int& vector_leading_dimension,
    threadgroup typename GEMVKernel<T, 8, 1, 1, 32, 4, 4, false, 1>::acc_type
        threadgroup_memory[GEMVKernel<T, 8, 1, 1, 32, 4, 4, false, 1>::tgp_mem_size == 0
            ? 1
            : GEMVKernel<T, 8, 1, 1, 32, 4, 4, false, 1>::tgp_mem_size],
    const bool apply_output_scale_and_accumulate SPECIALIZE,
    const uint threadgroup_index_x GROUPS((output_dimension + 32 - 1) / 32),
    const uint threadgroup_index_y GROUPS(batch_rows),
    const uint threadgroup_index_z GROUPS(batch_shape[0]),
    const uint thread_index_x THREADS(32),
    const uint thread_index_y THREADS(1),
    const uint thread_index_z THREADS(8),
    const Simd simd
) {
  if (batch_ndim == 0 || output_leading_dimension == 0 || vector_leading_dimension == 0) {
    return;
  }
  const uint3 threadgroup_position = uint3(threadgroup_index_x, threadgroup_index_y, threadgroup_index_z);
  const uint3 thread_position = uint3(thread_index_x, thread_index_y, thread_index_z);
  if (apply_output_scale_and_accumulate) {
    run_matmul_gemv_shape<T, 8, 1, 1, 32, 4, 4, true>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension,
        output_dimension,
        matrix_leading_dimension,
        output_scale,
        output_accumulate_scale,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride,
        batch_rows,
        threadgroup_memory,
        threadgroup_position,
        thread_position,
        simd.group_idx,
        simd.lane_idx
    );
  } else {
    run_matmul_gemv_shape<T, 8, 1, 1, 32, 4, 4, false>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension,
        output_dimension,
        matrix_leading_dimension,
        output_scale,
        output_accumulate_scale,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride,
        batch_rows,
        threadgroup_memory,
        threadgroup_position,
        thread_position,
        simd.group_idx,
        simd.lane_idx
    );
  }
}

