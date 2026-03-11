#pragma once

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
    const bool aligned_N,
    const int BM,
    const int BK,
    const int BN>
inline void quantized_gemm_transposed_core(
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
    if (!aligned_N && num_outputs < BN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        input_loader.load_checked(short2(BK, num_elements));
        weight_loader.load_checked(short2(BK, num_outputs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_operation.mma(input_shared, weight_shared);
        input_loader.next();
        weight_loader.next();
      }
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
    if (!aligned_N && num_outputs < BN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        input_loader.load_unchecked();
        weight_loader.load_checked(short2(BK, num_outputs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_operation.mma(input_shared, weight_shared);
        input_loader.next();
        weight_loader.next();
      }
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
  if (num_elements < BM || num_outputs < BN) {
    mma_operation.store_result_checked(y, N, short2(num_outputs, num_elements));
  } else {
    mma_operation.store_result(y, N);
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32,
    const bool use_affine_bias_quant = false>
void quantized_gemm_transposed_implementation(
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

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t =
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using input_loader_t =
      BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * 32>;

  const int k_packed = K * bytes_per_pack / pack_factor;
  const int k_group_count = (K + group_size - 1) / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto weight_bytes = (const device uint8_t*)w;

  const device T* x_block = x + y_row * static_cast<int64_t>(K);
  const device uint8_t* weight_block = weight_bytes + y_col * k_packed;
  scales += y_col * k_group_count;
  biases += y_col * k_group_count;
  device T* output_block = y + y_row * static_cast<int64_t>(N) + y_col;

  const short num_elements = min(BM, M - y_row);
  const short num_outputs = min(BN, N - y_col);
  input_loader_t input_loader(x_block, K, input_shared, simd_gid, simd_lid);
  mma_t mma_operation(simd_gid, simd_lid);

  if (use_affine_bias_quant) {
    using weight_loader_t = QuantizedBlockLoaderAffineBias<
        T,
        BN,
        BK,
        BK_padded,
        1,
        WM * WN * 32,
        group_size,
        bits>;
    weight_loader_t weight_loader(
        weight_block,
        scales,
        biases,
        K,
        weight_shared,
        simd_gid,
        simd_lid
    );
    quantized_gemm_transposed_core<
        weight_loader_t,
        input_loader_t,
        mma_t,
        T,
        aligned_N,
        BM,
        BK,
        BN>(
        input_loader,
        weight_loader,
        mma_operation,
        num_elements,
        num_outputs,
        K,
        output_block,
        N,
        input_shared,
        weight_shared,
        lid
    );
  } else {
    using weight_loader_t = QuantizedBlockLoaderZeroPoint<
        T,
        BN,
        BK,
        BK_padded,
        1,
        WM * WN * 32,
        group_size,
        bits>;
    const device uint8_t* column_zero_points =
        zero_points + y_col * (bits == 4 ? ((k_group_count + 1) / 2) : k_group_count);
    weight_loader_t weight_loader(
        weight_block,
        scales,
        column_zero_points,
        K,
        k_group_count,
        weight_shared,
        simd_gid,
        simd_lid
    );
    quantized_gemm_transposed_core<
        weight_loader_t,
        input_loader_t,
        mma_t,
        T,
        aligned_N,
        BM,
        BK,
        BN>(
        input_loader,
        weight_loader,
        mma_operation,
        num_elements,
        num_outputs,
        K,
        output_block,
        N,
        input_shared,
        weight_shared,
        lid
    );
  }
}
