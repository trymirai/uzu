#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

#include "../generated/quantization_method.h"
#include "../matmul/gemm/common/qmv_ops.h"
#include "../matmul/gemm/common/quant_pack.h"
#include "../matmul/gemm/common/quant_scale_bias.h"
#include "../matmul/gemm/common/quant_scale_zero_point.h"
#include "../matmul/gemm/common/quant_unpack.h"
#include "mma.h"

using QuantizationMethod = uzu::quantization_method::QuantizationMethod;

// Re-export the moved helpers/loaders at the legacy global scope so the
// standalone QMV/QMV-Fast/QMM kernels (`qmv.metal`, `qmv_fast.metal`,
// `qmm_transposed.metal`) and the `qmm_transposed_impl/_core` templates below
// compile unchanged. Step 6 removes these aliases together with the standalone
// QMM kernel.
using uzu::gemm::uint_to_fp;
using uzu::gemm::_uint4_to_fp4_float;
using uzu::gemm::uint4_to_fp4;
using uzu::gemm::get_pack_factor;
using uzu::gemm::get_bytes_per_pack;
using uzu::gemm::load_vector;
using uzu::gemm::load_vector_safe;
using uzu::gemm::qouter;
using uzu::gemm::qdot;
using uzu::gemm::qdot_safe;
using uzu::gemm::dequantize;
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
using QuantizedBlockLoaderScaleBias = uzu::gemm::QuantizedBlockLoaderScaleBias<
    T,
    BROWS,
    BCOLS,
    dst_ld,
    reduction_dim,
    tgp_size,
    group_size,
    bits>;
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits,
    bool per_output_layout = false>
using QuantizedBlockLoaderScaleZeroPoint = uzu::gemm::QuantizedBlockLoaderScaleZeroPoint<
    T,
    BROWS,
    BCOLS,
    dst_ld,
    reduction_dim,
    tgp_size,
    group_size,
    bits,
    per_output_layout>;

template <
    typename LoaderW,
    typename LoaderX,
    typename Mma,
    typename T,
    const bool aligned_N,
    const int BM,
    const int BK,
    const int BN>
inline void qmm_transposed_core(
    thread LoaderX& loader_x,
    thread LoaderW& loader_w,
    thread Mma& mma_op,
    const short num_els,
    const short num_outs,
    const int in_vec_size,
    device T* output,
    const int out_vec_size,
    threadgroup T* Xs,
    threadgroup T* Ws
) {
  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < in_vec_size; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < in_vec_size; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < in_vec_size; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < in_vec_size; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(output, out_vec_size, short2(num_outs, num_els));
  } else {
    mma_op.store_result(output, out_vec_size);
  }
}

template <
    typename T,
    const uint group_size,
    const uint bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32,
    const QuantizationMethod quant_method = QuantizationMethod::ScaleZeroPoint,
    const int WM = 2,
    const int WN = 2>
void qmm_transposed_impl(
    const device uint32_t* weights,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* input,
    device T* output,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const int in_vec_size,
    const int out_vec_size,
    const int batch_size,
    uint out_block_idx,
    uint batch_block_idx,
    uint simd_group,
    uint simd_lane
) {
  static_assert(BK >= 32, "BK should be larger than METAL_SIMD_SIZE");
  static_assert(BK % 32 == 0, "BK should be divisible by METAL_SIMD_SIZE");
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = matmul_utils::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      matmul_utils::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * 32>;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = (in_vec_size + group_size - 1) / group_size;
  const int out_row = batch_block_idx * BM;
  const int out_col = out_block_idx * BN;

  auto wl = (const device uint8_t*)weights;

  const device T* x_block = input + out_row * static_cast<int64_t>(in_vec_size);
  const device uint8_t* w_block = wl + out_col * in_vec_size_w;
  scales += out_col * in_vec_size_g;
  biases += out_col * in_vec_size_g;
  device T* y_block =
      output + out_row * static_cast<int64_t>(out_vec_size) + out_col;

  const short num_els = min(BM, batch_size - out_row);
  const short num_outs = min(BN, out_vec_size - out_col);
  loader_x_t loader_x(x_block, in_vec_size, Xs, simd_group, simd_lane);
  mma_t mma_op(simd_group, simd_lane);

  if (quant_method == QuantizationMethod::ScaleBias) {
    using loader_w_t = QuantizedBlockLoaderScaleBias<
        T,
        BN,
        BK,
        BK_padded,
        1,
        WM * WN * 32,
        group_size,
        bits>;
    loader_w_t loader_w(
        w_block,
        scales,
        biases,
        in_vec_size,
        Ws,
        simd_group,
        simd_lane
    );
    qmm_transposed_core<
        loader_w_t,
        loader_x_t,
        mma_t,
        T,
        aligned_N,
        BM,
        BK,
        BN>(
        loader_x,
        loader_w,
        mma_op,
        num_els,
        num_outs,
        in_vec_size,
        y_block,
        out_vec_size,
        Xs,
        Ws
    );
  } else {
    using loader_w_t = QuantizedBlockLoaderScaleZeroPoint<
        T,
        BN,
        BK,
        BK_padded,
        1,
        WM * WN * 32,
        group_size,
        bits>;
    const device uint8_t* zero_points_row =
        zero_points +
        out_col * (bits == 4 ? ((in_vec_size_g + 1) / 2) : in_vec_size_g);
    loader_w_t loader_w(
        w_block,
        scales,
        zero_points_row,
        in_vec_size,
        in_vec_size_g,
        Ws,
        simd_group,
        simd_lane
    );
    qmm_transposed_core<
        loader_w_t,
        loader_x_t,
        mma_t,
        T,
        aligned_N,
        BM,
        BK,
        BN>(
        loader_x,
        loader_w,
        mma_op,
        num_els,
        num_outs,
        in_vec_size,
        y_block,
        out_vec_size,
        Xs,
        Ws
    );
  }
}
