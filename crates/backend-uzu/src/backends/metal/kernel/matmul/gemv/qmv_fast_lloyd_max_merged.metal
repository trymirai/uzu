#include <metal_stdlib>
#include "../../common/dsl.h"
#include "../../hadamard_transform/hadamard_transform.h"
#include "../../generated/quantization_method.h"
#include "../common/qdot.h"
#include "../gemm/common/quant_pack.h"

using namespace uzu::quantization_method;
using namespace uzu::gemm;

#define NUM_SIMDGROUPS 8
#define BIAS_CODEBOOK_SIZE 16

template <typename T, uint GROUP_SIZE, uint BITS>
VARIANTS(T, float, half, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4)
PUBLIC KERNEL(QuantizedMatmulQmvFastLloydMaxMerged)(
    const device uint32_t* weights,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(quant_method == QuantizationMethod::ScaleZeroPoint),
    const device T* biases OPTIONAL(quant_method == QuantizationMethod::ScaleBias),
    const device half* codebook OPTIONAL(quant_method == QuantizationMethod::LloydMax),
    const device uint8_t* bias_indices OPTIONAL(quant_method == QuantizationMethod::LloydMax),
    const device half* bias_codebook OPTIONAL(quant_method == QuantizationMethod::LloydMax),
    const device T* input,
    device T* output,
    const device int32_t* hadamard_factors OPTIONAL(use_hadamard),
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    const QuantizationMethod quant_method SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    threadgroup half codebook_tg[NUM_SIMDGROUPS][1 << BITS],
    threadgroup half bias_codebook_tg[NUM_SIMDGROUPS][BIAS_CODEBOOK_SIZE],
    const uint batch_idx GROUPS(batch_size),
    const uint out_block_idx GROUPS(out_vec_size.div_ceil(32)),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(NUM_SIMDGROUPS)
) {
  constexpr uint packs_per_thread = BITS == 2 ? 1 : 2;
  constexpr uint results_per_simdgroup = 4;
  constexpr uint pack_factor = get_pack_factor<BITS, 32>();
  constexpr uint bytes_per_pack = get_bytes_per_pack<BITS, 32>();
  constexpr uint values_per_thread = pack_factor * packs_per_thread;
  constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
  constexpr uint scale_step_per_thread = GROUP_SIZE / values_per_thread;
  constexpr uint codebook_size = 1u << BITS;

  const device uint8_t* ws = (const device uint8_t*)weights;
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const uint in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const uint in_vec_size_g = in_vec_size / GROUP_SIZE;
  const uint out_row =
      out_block_idx * (NUM_SIMDGROUPS * results_per_simdgroup) +
      simd_group * results_per_simdgroup;
  ws += out_row * in_vec_size_w + simd_lane * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lane / scale_step_per_thread;

  uint zp_stride = 0;
  const device uint8_t* zps = nullptr;
  bool high_nibble = false;
  uint bias_stride = 0;
  const device uint8_t* bias_index_ptr = nullptr;
  bool bias_high_nibble = false;

  threadgroup half* codebook_sg = codebook_tg[simd_group];
  threadgroup half* bias_codebook_sg = bias_codebook_tg[simd_group];

  if (quant_method == QuantizationMethod::LloydMax) {
    static_assert(BITS == 4, "Only int4 Lloyd-Max QMV is supported");
    bias_stride = (in_vec_size_g + 1) / 2;
    uint bias_g_offset = simd_lane / scale_step_per_thread;
    bias_index_ptr = bias_indices + out_row * bias_stride + bias_g_offset / 2;
    bias_high_nibble = (bias_g_offset & 1);
    static_assert(
        (block_size / GROUP_SIZE) % 2 == 0,
        "bias_high_nibble parity must be preserved across blocks"
    );

    for (uint entry = simd_lane; entry < codebook_size;
         entry += METAL_SIMD_SIZE) {
      codebook_sg[entry] = codebook[entry];
    }
    for (uint entry = simd_lane; entry < BIAS_CODEBOOK_SIZE;
         entry += METAL_SIMD_SIZE) {
      bias_codebook_sg[entry] = bias_codebook[entry];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  } else if (quant_method == QuantizationMethod::ScaleBias) {
    biases += out_row * in_vec_size_g + simd_lane / scale_step_per_thread;
  } else if (quant_method == QuantizationMethod::ScaleZeroPoint) {
    if (BITS == 4) {
      zp_stride = (in_vec_size_g + 1) / 2;
      zps = zero_points + out_row * zp_stride;
      uint g_offset = simd_lane / scale_step_per_thread;
      zps += g_offset / 2;
      high_nibble = (g_offset & 1);
    } else {
      zp_stride = in_vec_size_g;
      zps = zero_points + out_row * zp_stride;
      zps += simd_lane / scale_step_per_thread;
    }
  }

  input += batch_idx * in_vec_size + simd_lane * values_per_thread;
  output += batch_idx * out_vec_size + out_row;

  for (uint k = 0; k < in_vec_size; k += block_size) {
    // quant_method is a SPECIALIZE constant: the unused sum on the LloydMax
    // path is dropped at PSO compile, matching the standalone kernel exactly.
    U sum = 0;
    if (quant_method == QuantizationMethod::LloydMax) {
      load_vector_unscaled<T, U, values_per_thread>(input, x_thread);
    } else {
      sum = load_vector<T, U, values_per_thread, BITS>(input, x_thread);
    }

    auto wl0 = (const device uint8_t*)(ws);
    auto wl1 = (const device uint8_t*)(ws + in_vec_size_w);
    auto wl2 = (const device uint8_t*)(ws + 2 * in_vec_size_w);
    auto wl3 = (const device uint8_t*)(ws + 3 * in_vec_size_w);

    U s0 = static_cast<U>(scales[0]);
    U s1 = static_cast<U>(scales[in_vec_size_g]);
    U s2 = static_cast<U>(scales[2 * in_vec_size_g]);
    U s3 = static_cast<U>(scales[3 * in_vec_size_g]);

    if (quant_method == QuantizationMethod::LloydMax) {
      uchar4 bias_bytes = uchar4(
          bias_index_ptr[0],
          bias_index_ptr[bias_stride],
          bias_index_ptr[2 * bias_stride],
          bias_index_ptr[3 * bias_stride]
      );
      const uint8_t shift = bias_high_nibble ? 4u : 0u;
      uchar4 bias_codes = (bias_bytes >> shift) & uchar4(0x0F);

      U b0 = static_cast<U>(bias_codebook_sg[bias_codes.x]);
      U b1 = static_cast<U>(bias_codebook_sg[bias_codes.y]);
      U b2 = static_cast<U>(bias_codebook_sg[bias_codes.z]);
      U b3 = static_cast<U>(bias_codebook_sg[bias_codes.w]);

      result[0] += qdot_lloyd_max<U, values_per_thread, BITS>(
          wl0,
          x_thread,
          codebook_sg,
          s0,
          b0
      );
      result[1] += qdot_lloyd_max<U, values_per_thread, BITS>(
          wl1,
          x_thread,
          codebook_sg,
          s1,
          b1
      );
      result[2] += qdot_lloyd_max<U, values_per_thread, BITS>(
          wl2,
          x_thread,
          codebook_sg,
          s2,
          b2
      );
      result[3] += qdot_lloyd_max<U, values_per_thread, BITS>(
          wl3,
          x_thread,
          codebook_sg,
          s3,
          b3
      );
    } else if (quant_method == QuantizationMethod::ScaleBias) {
      U b0 = static_cast<U>(biases[0]);
      U b1 = static_cast<U>(biases[in_vec_size_g]);
      U b2 = static_cast<U>(biases[2 * in_vec_size_g]);
      U b3 = static_cast<U>(biases[3 * in_vec_size_g]);
      result[0] += qdot<U, values_per_thread, BITS>(wl0, x_thread, s0, b0, sum);
      result[1] += qdot<U, values_per_thread, BITS>(wl1, x_thread, s1, b1, sum);
      result[2] += qdot<U, values_per_thread, BITS>(wl2, x_thread, s2, b2, sum);
      result[3] += qdot<U, values_per_thread, BITS>(wl3, x_thread, s3, b3, sum);
    } else {
      uchar4 zp_bytes = uchar4(
          zps[0],
          zps[zp_stride],
          zps[2 * zp_stride],
          zps[3 * zp_stride]
      );
      uchar4 zp_nibbles;
      if (BITS == 4) {
        const uint8_t shift = high_nibble ? 4u : 0u;
        zp_nibbles = (zp_bytes >> shift) & uchar4(0x0F);
      } else {
        zp_nibbles = zp_bytes;
      }
      result[0] += qdot<U, values_per_thread, BITS>(
          wl0,
          x_thread,
          s0,
          -s0 * static_cast<U>(zp_nibbles.x),
          sum
      );
      result[1] += qdot<U, values_per_thread, BITS>(
          wl1,
          x_thread,
          s1,
          -s1 * static_cast<U>(zp_nibbles.y),
          sum
      );
      result[2] += qdot<U, values_per_thread, BITS>(
          wl2,
          x_thread,
          s2,
          -s2 * static_cast<U>(zp_nibbles.z),
          sum
      );
      result[3] += qdot<U, values_per_thread, BITS>(
          wl3,
          x_thread,
          s3,
          -s3 * static_cast<U>(zp_nibbles.w),
          sum
      );
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / GROUP_SIZE;
    if (quant_method == QuantizationMethod::LloydMax) {
      bias_index_ptr += (block_size / GROUP_SIZE) / 2;
    } else if (quant_method == QuantizationMethod::ScaleBias) {
      biases += block_size / GROUP_SIZE;
    } else if (quant_method == QuantizationMethod::ScaleZeroPoint) {
      if (BITS == 4) {
        zps += (block_size / GROUP_SIZE) / 2;
      } else {
        zps += block_size / GROUP_SIZE;
      }
    }
    input += block_size;
  }

  for (uint row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
  }

  if (use_hadamard) {
    if (simd_lane == 0) {
      for (uint row = 0; row < results_per_simdgroup; row++) {
        shared_results[simd_group * results_per_simdgroup + row] = result[row];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
      uint global_out_idx = out_block_idx * 32 + simd_lane;
      if (global_out_idx < out_vec_size) {
        output[simd_lane] = simdgroup_output_random_hadamard_transform(
            static_cast<ushort>(simd_lane),
            T(shared_results[simd_lane]),
            hadamard_factors[global_out_idx]
        );
      }
    }
  } else {
    qmv_write_direct_results<T, U, results_per_simdgroup>(
        result,
        output,
        simd_lane
    );
  }
}
