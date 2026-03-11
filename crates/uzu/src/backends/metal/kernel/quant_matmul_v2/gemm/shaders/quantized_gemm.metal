#include <metal_stdlib>
#include "../../../definitions.metal"
#include "../../common/quantized_loader.h"

using namespace uzu::quantized_matmul;
using namespace uzu::matmul;

template <
    typename WeightLoaderType,
    typename InputLoaderType,
    typename MmaType,
    typename T,
    const bool aligned_K,
    const int BM,
    const int BK,
    const int BN>
inline void quantized_gemm_core(
    thread InputLoaderType& input_loader,
    thread WeightLoaderType& weight_loader,
    thread MmaType& mma_operation,
    const short num_elements,
    const short num_outputs,
    const int K,
    device T* y,
    const constant int& N,
    threadgroup T* input_shared,
    threadgroup T* weight_shared,
    uint lid [[thread_index_in_threadgroup]]
) {
  if (num_elements < BM) {
    if ((K % BK) != 0) {
      const int k_block_count = K / BK;
      for (int k = 0; k < k_block_count; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        input_loader.load_checked(short2(BK, num_elements));
        weight_loader.load_unchecked();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_operation.mma(input_shared, weight_shared);
        input_loader.next();
        weight_loader.next();
      }
      const short remaining_k = K - k_block_count * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      input_loader.load_checked(short2(remaining_k, num_elements));
      weight_loader.load_checked(short2(BN, remaining_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_operation.mma(input_shared, weight_shared);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        input_loader.load_checked(short2(BK, num_elements));
        weight_loader.load_unchecked();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_operation.mma(input_shared, weight_shared);
        input_loader.next();
        weight_loader.next();
      }
    }
  } else {
    if ((K % BK) != 0) {
      const int k_block_count = K / BK;
      for (int k = 0; k < k_block_count; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        input_loader.load_unchecked();
        weight_loader.load_unchecked();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_operation.mma(input_shared, weight_shared);
        input_loader.next();
        weight_loader.next();
      }
      const short remaining_k = K - k_block_count * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      input_loader.load_checked(short2(remaining_k, BM));
      weight_loader.load_checked(short2(BN, remaining_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_operation.mma(input_shared, weight_shared);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        input_loader.load_unchecked();
        weight_loader.load_unchecked();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_operation.mma(input_shared, weight_shared);
        input_loader.next();
        weight_loader.next();
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_elements < BM) {
    mma_operation.store_result_checked(y, N, short2(BN, num_elements));
  } else {
    mma_operation.store_result(y, N);
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_K = false,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32,
    const bool use_affine_bias_quant = false>
void quantized_gemm_implementation(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* input_shared,
    threadgroup T* weight_shared,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  static_assert(BK >= 32, "BK should be larger than SIMD_SIZE");
  static_assert(BK % 32 == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t =
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, false, BK_padded, BN_padded>;
  using input_loader_t =
      BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * 32, 1, 4>;

  auto weight_bytes = (const device uint8_t*)w;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * static_cast<int64_t>(K);
  weight_bytes += y_col * bytes_per_pack / pack_factor;
  y += y_row * static_cast<int64_t>(N) + y_col;

  const device T* scales_base = scales;
  const device T* biases_base = biases;

  const short num_elements = min(BM, M - y_row);
  const short num_outputs = min(BN, N - y_col);
  input_loader_t input_loader(x, K, input_shared, simd_gid, simd_lid);
  mma_t mma_operation(simd_gid, simd_lid);

  if (use_affine_bias_quant) {
    using weight_loader_t = QuantizedBlockLoaderAffineBias<
        T,
        BK,
        BN,
        BN_padded,
        0,
        WM * WN * 32,
        group_size,
        bits>;
    const device T* column_scales = scales_base + (y_col / group_size);
    const device T* column_biases = biases_base + (y_col / group_size);
    weight_loader_t weight_loader(
        weight_bytes,
        column_scales,
        column_biases,
        N,
        weight_shared,
        simd_gid,
        simd_lid
    );
    quantized_gemm_core<
        weight_loader_t,
        input_loader_t,
        mma_t,
        T,
        aligned_K,
        BM,
        BK,
        BN>(
        input_loader,
        weight_loader,
        mma_operation,
        num_elements,
        num_outputs,
        K,
        y,
        N,
        input_shared,
        weight_shared,
        lid
    );
  } else {
    const int output_groups_total = (N + group_size - 1) / group_size;
    const int groups_per_row = output_groups_total;
    const int output_group = y_col / group_size;
    const int zero_point_stride_output =
        (bits == 4) ? ((output_groups_total + 1) / 2) : output_groups_total;
    using weight_loader_t = QuantizedBlockLoaderZeroPoint<
        T,
        BK,
        BN,
        BN_padded,
        0,
        WM * WN * 32,
        group_size,
        bits,
        true>;
    weight_loader_t weight_loader(
        weight_bytes,
        scales_base,
        zero_points,
        N,
        groups_per_row,
        weight_shared,
        simd_gid,
        simd_lid,
        output_group,
        output_groups_total,
        zero_point_stride_output
    );
    quantized_gemm_core<
        weight_loader_t,
        input_loader_t,
        mma_t,
        T,
        aligned_K,
        BM,
        BK,
        BN>(
        input_loader,
        weight_loader,
        mma_operation,
        num_elements,
        num_outputs,
        K,
        y,
        N,
        input_shared,
        weight_shared,
        lid
    );
  }
}

template <typename T, int GROUP_SIZE, int BITS>
VARIANTS(T, float, half, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4, 8)
KERNEL(QuantizedMatmulGemmV2)(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device T* x,
    device T* y,
    const constant int& k,
    const constant int& n,
    const constant int& m,
    threadgroup T input_shared[32 * (32 + 16 / sizeof(T))],
    threadgroup T weight_shared[32 * (32 + 16 / sizeof(T))],
    const bool use_zero_points SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    const bool aligned_k SPECIALIZE,
    const uint tgid_x GROUPS((n + 32 - 1) / 32),
    const uint tgid_y GROUPS((m + 32 - 1) / 32),
    const uint tgid_z GROUPS(1),
    const uint tid_x THREADS(32),
    const uint tid_y THREADS(2),
    const uint tid_z THREADS(2)
) {
  const uint3 tid = uint3(tgid_x, tgid_y, tgid_z);
  const uint lid = tid_z * 64 + tid_y * 32 + tid_x;
  const uint simd_gid = tid_z * 2 + tid_y;
  const uint simd_lid = tid_x;

  if (use_mlx_quant) {
    if (aligned_k) {
      quantized_gemm_implementation<T, GROUP_SIZE, BITS, true, 32, 32, 32, true>(
          w,
          scales,
          zero_points,
          biases,
          x,
          y,
          input_shared,
          weight_shared,
          k,
          n,
          m,
          tid,
          lid,
          simd_gid,
          simd_lid
      );
    } else {
      quantized_gemm_implementation<T, GROUP_SIZE, BITS, false, 32, 32, 32, true>(
          w,
          scales,
          zero_points,
          biases,
          x,
          y,
          input_shared,
          weight_shared,
          k,
          n,
          m,
          tid,
          lid,
          simd_gid,
          simd_lid
      );
    }
  } else {
    if (aligned_k) {
      quantized_gemm_implementation<T, GROUP_SIZE, BITS, true, 32, 32, 32, false>(
          w,
          scales,
          zero_points,
          biases,
          x,
          y,
          input_shared,
          weight_shared,
          k,
          n,
          m,
          tid,
          lid,
          simd_gid,
          simd_lid
      );
    } else {
      quantized_gemm_implementation<T, GROUP_SIZE, BITS, false, 32, 32, 32, false>(
          w,
          scales,
          zero_points,
          biases,
          x,
          y,
          input_shared,
          weight_shared,
          k,
          n,
          m,
          tid,
          lid,
          simd_gid,
          simd_lid
      );
    }
  }
}
