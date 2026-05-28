#pragma once

// Out-of-class definition of GemvCore::run_quantized, the group-quantized
// reduction body. Designed to be included via gemv_core.h (which declares
// GemvCore, the `qmv_*` helpers, and the shared using-namespace directives);
// not intended to be included standalone.

namespace uzu {
namespace gemv {

// Quantized GEMV reduction. Called from GemvCore::run when BITS != 0. Layout is
// hardcoded (8 simdgroups × 4 results) — the host pins the canonical tile so
// the per-simdgroup constants compile to immediates here. The two quant
// sub-kinds (ScaleBias vs ScaleZeroPoint) branch on the runtime `quant_method`
// function constant.
template <typename T, uint GROUP_SIZE, uint BITS>
METAL_FUNC void GemvCore<T, GROUP_SIZE, BITS>::run_quantized(
    const device uint32_t* weights,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* input,
    device T* output,
    const device int32_t* hadamard_factors,
    const device T* output_bias,
    const constant uzu::matmul::GemvParams* params,
    QuantizationMethod quant_method,
    GemmDTransform output_transform,
    uint group_index_x,
    uint group_index_y,
    uint thread_index_x,
    uint thread_index_y,
    threadgroup float* shared_results
) {
  const uint in_vec_size = params->in_vec_size;
  const uint out_vec_size = params->out_vec_size;
  const float output_scale = params->ab_scale;
  const bool is_scale = output_transform.contains(GemmDTransform::SCALE);
  const bool is_accumulate =
      output_transform.contains(GemmDTransform::ACCUMULATE);
  const bool is_bias = output_transform.contains(GemmDTransform::BIAS);
  const bool use_hadamard = output_transform.contains(GemmDTransform::RHT);

  const uint simd_lane = thread_index_x;
  const uint simd_group = thread_index_y;
  const uint batch_idx = group_index_x;
  const uint out_block_idx = group_index_y;

  constexpr uint packs_per_thread = BITS == 2 ? 1 : 2;
  constexpr uint num_simdgroups = 8;
  constexpr uint results_per_simdgroup = 4;
  constexpr uint pack_factor = qmv_pack_factor<BITS>();
  constexpr uint bytes_per_pack = qmv_bytes_per_pack<BITS>();
  constexpr uint values_per_thread = pack_factor * packs_per_thread;
  constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
  constexpr uint scale_step_per_thread = GROUP_SIZE / values_per_thread;
  const device uint8_t* ws = (const device uint8_t*)weights;
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const uint in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const uint in_vec_size_g = in_vec_size / GROUP_SIZE;
  const uint out_row =
      out_block_idx * (num_simdgroups * results_per_simdgroup) +
      simd_group * results_per_simdgroup;
  ws += out_row * in_vec_size_w + simd_lane * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lane / scale_step_per_thread;

  uint zp_stride = 0;
  const device uint8_t* zps = nullptr;
  bool high_nibble = false;

  if (quant_method == QuantizationMethod::ScaleBias) {
    biases += out_row * in_vec_size_g + simd_lane / scale_step_per_thread;
  } else {
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

  uint k = 0;
  for (; k + block_size <= in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, BITS>(input, x_thread);

    {
      auto wl0 = (const device uint8_t*)(ws);
      auto wl1 = (const device uint8_t*)(ws + in_vec_size_w);
      auto wl2 = (const device uint8_t*)(ws + 2 * in_vec_size_w);
      auto wl3 = (const device uint8_t*)(ws + 3 * in_vec_size_w);

      U s0 = static_cast<U>(scales[0]);
      U s1 = static_cast<U>(scales[in_vec_size_g]);
      U s2 = static_cast<U>(scales[2 * in_vec_size_g]);
      U s3 = static_cast<U>(scales[3 * in_vec_size_g]);

      if (quant_method == QuantizationMethod::ScaleBias) {
        U b0 = static_cast<U>(biases[0]);
        U b1 = static_cast<U>(biases[in_vec_size_g]);
        U b2 = static_cast<U>(biases[2 * in_vec_size_g]);
        U b3 = static_cast<U>(biases[3 * in_vec_size_g]);
        result[0] +=
            qdot<U, values_per_thread, BITS>(wl0, x_thread, s0, b0, sum);
        result[1] +=
            qdot<U, values_per_thread, BITS>(wl1, x_thread, s1, b1, sum);
        result[2] +=
            qdot<U, values_per_thread, BITS>(wl2, x_thread, s2, b2, sum);
        result[3] +=
            qdot<U, values_per_thread, BITS>(wl3, x_thread, s3, b3, sum);
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
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / GROUP_SIZE;
    if (quant_method == QuantizationMethod::ScaleBias) {
      biases += block_size / GROUP_SIZE;
    } else {
      if (BITS == 4) {
        zps += (block_size / GROUP_SIZE) / 2;
      } else {
        zps += block_size / GROUP_SIZE;
      }
    }
    input += block_size;
  }

  const uint thread_offset = simd_lane * values_per_thread;
  const int remaining =
      (k + thread_offset < in_vec_size)
          ? min(int(in_vec_size - k - thread_offset), int(values_per_thread))
          : 0;
  if (remaining > 0) {
    U sum = load_vector_safe<T, U, values_per_thread, BITS>(
        input,
        x_thread,
        remaining
    );

    auto wl0 = (const device uint8_t*)(ws);
    auto wl1 = (const device uint8_t*)(ws + in_vec_size_w);
    auto wl2 = (const device uint8_t*)(ws + 2 * in_vec_size_w);
    auto wl3 = (const device uint8_t*)(ws + 3 * in_vec_size_w);

    U s0 = static_cast<U>(scales[0]);
    U s1 = static_cast<U>(scales[in_vec_size_g]);
    U s2 = static_cast<U>(scales[2 * in_vec_size_g]);
    U s3 = static_cast<U>(scales[3 * in_vec_size_g]);

    if (quant_method == QuantizationMethod::ScaleBias) {
      U b0 = static_cast<U>(biases[0]);
      U b1 = static_cast<U>(biases[in_vec_size_g]);
      U b2 = static_cast<U>(biases[2 * in_vec_size_g]);
      U b3 = static_cast<U>(biases[3 * in_vec_size_g]);
      result[0] += qdot_safe<U, values_per_thread, BITS>(
          wl0,
          x_thread,
          s0,
          b0,
          sum,
          remaining
      );
      result[1] += qdot_safe<U, values_per_thread, BITS>(
          wl1,
          x_thread,
          s1,
          b1,
          sum,
          remaining
      );
      result[2] += qdot_safe<U, values_per_thread, BITS>(
          wl2,
          x_thread,
          s2,
          b2,
          sum,
          remaining
      );
      result[3] += qdot_safe<U, values_per_thread, BITS>(
          wl3,
          x_thread,
          s3,
          b3,
          sum,
          remaining
      );
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
      result[0] += qdot_safe<U, values_per_thread, BITS>(
          wl0,
          x_thread,
          s0,
          -s0 * static_cast<U>(zp_nibbles.x),
          sum,
          remaining
      );
      result[1] += qdot_safe<U, values_per_thread, BITS>(
          wl1,
          x_thread,
          s1,
          -s1 * static_cast<U>(zp_nibbles.y),
          sum,
          remaining
      );
      result[2] += qdot_safe<U, values_per_thread, BITS>(
          wl2,
          x_thread,
          s2,
          -s2 * static_cast<U>(zp_nibbles.z),
          sum,
          remaining
      );
      result[3] += qdot_safe<U, values_per_thread, BITS>(
          wl3,
          x_thread,
          s3,
          -s3 * static_cast<U>(zp_nibbles.w),
          sum,
          remaining
      );
    }
  }

  for (uint row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
  }

  // Fused output epilogue: optional scale, then optional accumulate / bias.
  if (simd_lane == 0) {
    for (uint row = 0; row < results_per_simdgroup; row++) {
      U value = result[row];
      if (is_scale) {
        value = static_cast<U>(output_scale) * value;
      }
      const uint global_row = out_row + row;
      if (is_accumulate && global_row < out_vec_size) {
        value += static_cast<U>(output[row]);
      }
      if (is_bias && global_row < out_vec_size) {
        value += static_cast<U>(output_bias[global_row]);
      }
      result[row] = value;
    }
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
    if (simd_lane == 0) {
      for (uint row = 0; row < results_per_simdgroup; row++) {
        output[row] = static_cast<T>(result[row]);
      }
    }
  }
}

} // namespace gemv
} // namespace uzu
