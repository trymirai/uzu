#include <metal_stdlib>
#include "../../common/dsl.h"
#include "../../hadamard_transform/hadamard_transform.h"
#include "../../generated/quantization_method.h"
#include "../../generated/gemm.h"
#include "../common/qdot.h"
#include "../common/quant_pack.h"

using namespace metal;
using namespace uzu::quantization_method;
using namespace uzu::gemm;

template <
    typename T,
    GemmBPrologueKind B_PROLOGUE,
    uint GROUP_SIZE,
    uint BITS,
    uint K_SPLIT>
VARIANTS(T, float, half, bfloat)
VARIANTS(
    B_PROLOGUE,
    GemmBPrologueKind::FullPrecision,
    GemmBPrologueKind::ScaleBiasDequant,
    GemmBPrologueKind::ScaleZeroPointDequant)
VARIANTS(GROUP_SIZE, 0, 32, 64, 128)
VARIANTS(BITS, 0, 4, 8)
VARIANTS(K_SPLIT, 1, 2, 4, 8)
CONSTRAINT((B_PROLOGUE == GemmBPrologueKind::FullPrecision) == (BITS == 0))
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
CONSTRAINT(B_PROLOGUE == GemmBPrologueKind::FullPrecision || K_SPLIT == 1)
KERNEL(Gemv)(
    const device uint32_t* weights,
    const device T* scales
        OPTIONAL(B_PROLOGUE != GemmBPrologueKind::FullPrecision),
    const device uint8_t* zero_points
        OPTIONAL(B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant),
    const device T* biases
        OPTIONAL(B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant),
    const device T* input,
    device T* output,
    const device T* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* hadamard_factors
        OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    const constant float& ab_scale,
    const constant uint& group_count_x,
    const GemmDTransform output_transform SPECIALIZE,
    const bool input_aligned SPECIALIZE,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    const uint batch_idx GROUPS(batch_size),
    const uint out_block_idx GROUPS(group_count_x),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(8)
) {
  const bool is_scale = output_transform.contains(GemmDTransform::SCALE);
  const bool is_accumulate =
      output_transform.contains(GemmDTransform::ACCUMULATE);
  const bool is_bias = output_transform.contains(GemmDTransform::BIAS);
  const bool use_hadamard = output_transform.contains(GemmDTransform::RHT);

  constexpr uint num_simdgroups = 8;
  constexpr uint results_per_simdgroup = 4;
  typedef float U;
  thread U result[results_per_simdgroup] = {0};

  constexpr uint k_split = K_SPLIT;
  const uint row_group = simd_group / k_split;
  const uint k_slice = simd_group % k_split;
  constexpr uint rows_per_threadgroup = (num_simdgroups / k_split) * results_per_simdgroup;
  const uint out_row =
      out_block_idx * rows_per_threadgroup + row_group * results_per_simdgroup;
  output += batch_idx * out_vec_size + out_row;

  if constexpr (B_PROLOGUE == GemmBPrologueKind::FullPrecision) {
    constexpr uint values_per_thread = 4;
    constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
    typedef vec<T, 4> T4;
    const uint k_stride = k_split * block_size;
    const uint k_start = k_slice * block_size;
    const device T* w = reinterpret_cast<const device T*>(weights);
    w += out_row * in_vec_size + simd_lane * values_per_thread + k_start;
    const device T* in = input + batch_idx * in_vec_size +
        simd_lane * values_per_thread + k_start;

    uint k = k_start;
    for (; k + block_size <= in_vec_size; k += k_stride) {
      float4 xv = static_cast<float4>(*reinterpret_cast<const device T4*>(in));
      result[0] += dot(
          static_cast<float4>(*reinterpret_cast<const device T4*>(w)), xv);
      result[1] += dot(
          static_cast<float4>(
              *reinterpret_cast<const device T4*>(w + in_vec_size)),
          xv);
      result[2] += dot(
          static_cast<float4>(
              *reinterpret_cast<const device T4*>(w + 2 * in_vec_size)),
          xv);
      result[3] += dot(
          static_cast<float4>(
              *reinterpret_cast<const device T4*>(w + 3 * in_vec_size)),
          xv);
      w += k_stride;
      in += k_stride;
    }

    const uint thread_offset = simd_lane * values_per_thread;
    const int remaining =
        (k_split == 1 && !input_aligned && k + thread_offset < in_vec_size)
        ? min(static_cast<int>(in_vec_size - k - thread_offset),
              static_cast<int>(values_per_thread))
        : 0;
    if (remaining > 0) {
      const device T* w0 = w;
      const device T* w1 = w + in_vec_size;
      const device T* w2 = w + 2 * in_vec_size;
      const device T* w3 = w + 3 * in_vec_size;
      for (int j = 0; j < remaining; j++) {
        U x = static_cast<U>(in[j]);
        result[0] += static_cast<U>(w0[j]) * x;
        result[1] += static_cast<U>(w1[j]) * x;
        result[2] += static_cast<U>(w2[j]) * x;
        result[3] += static_cast<U>(w3[j]) * x;
      }
    }
  } else {
    constexpr uint packs_per_thread = BITS == 2 ? 1 : 2;
    constexpr uint pack_factor = get_pack_factor<BITS, 32>();
    constexpr uint bytes_per_pack = get_bytes_per_pack<BITS, 32>();
    constexpr uint values_per_thread = pack_factor * packs_per_thread;
    constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
    constexpr uint scale_step_per_thread = GROUP_SIZE / values_per_thread;
    const device uint8_t* ws = (const device uint8_t*)weights;
    thread U x_thread[values_per_thread];

    const uint in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const uint in_vec_size_g = in_vec_size / GROUP_SIZE;
    ws += out_row * in_vec_size_w +
        simd_lane * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lane / scale_step_per_thread;

    uint zp_stride = 0;
    const device uint8_t* zps = nullptr;
    bool high_nibble = false;

    if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
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

    const device T* in =
        input + batch_idx * in_vec_size + simd_lane * values_per_thread;

    uint k = 0;
    for (; k + block_size <= in_vec_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, BITS>(in, x_thread);

      {
        auto wl0 = (const device uint8_t*)(ws);
        auto wl1 = (const device uint8_t*)(ws + in_vec_size_w);
        auto wl2 = (const device uint8_t*)(ws + 2 * in_vec_size_w);
        auto wl3 = (const device uint8_t*)(ws + 3 * in_vec_size_w);

        U s0 = static_cast<U>(scales[0]);
        U s1 = static_cast<U>(scales[in_vec_size_g]);
        U s2 = static_cast<U>(scales[2 * in_vec_size_g]);
        U s3 = static_cast<U>(scales[3 * in_vec_size_g]);

        if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
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
              wl0, x_thread, s0, -s0 * static_cast<U>(zp_nibbles.x), sum);
          result[1] += qdot<U, values_per_thread, BITS>(
              wl1, x_thread, s1, -s1 * static_cast<U>(zp_nibbles.y), sum);
          result[2] += qdot<U, values_per_thread, BITS>(
              wl2, x_thread, s2, -s2 * static_cast<U>(zp_nibbles.z), sum);
          result[3] += qdot<U, values_per_thread, BITS>(
              wl3, x_thread, s3, -s3 * static_cast<U>(zp_nibbles.w), sum);
        }
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / GROUP_SIZE;
      if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
        biases += block_size / GROUP_SIZE;
      } else {
        if (BITS == 4) {
          zps += (block_size / GROUP_SIZE) / 2;
        } else {
          zps += block_size / GROUP_SIZE;
        }
      }
      in += block_size;
    }

    const uint thread_offset = simd_lane * values_per_thread;
    const int remaining = (!input_aligned && k + thread_offset < in_vec_size)
        ? min(static_cast<int>(in_vec_size - k - thread_offset),
              static_cast<int>(values_per_thread))
        : 0;
    if (remaining > 0) {
      U sum =
          load_vector_safe<T, U, values_per_thread, BITS>(in, x_thread, remaining);

      auto wl0 = (const device uint8_t*)(ws);
      auto wl1 = (const device uint8_t*)(ws + in_vec_size_w);
      auto wl2 = (const device uint8_t*)(ws + 2 * in_vec_size_w);
      auto wl3 = (const device uint8_t*)(ws + 3 * in_vec_size_w);

      U s0 = static_cast<U>(scales[0]);
      U s1 = static_cast<U>(scales[in_vec_size_g]);
      U s2 = static_cast<U>(scales[2 * in_vec_size_g]);
      U s3 = static_cast<U>(scales[3 * in_vec_size_g]);

      if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
        U b0 = static_cast<U>(biases[0]);
        U b1 = static_cast<U>(biases[in_vec_size_g]);
        U b2 = static_cast<U>(biases[2 * in_vec_size_g]);
        U b3 = static_cast<U>(biases[3 * in_vec_size_g]);
        result[0] += qdot_safe<U, values_per_thread, BITS>(
            wl0, x_thread, s0, b0, sum, remaining);
        result[1] += qdot_safe<U, values_per_thread, BITS>(
            wl1, x_thread, s1, b1, sum, remaining);
        result[2] += qdot_safe<U, values_per_thread, BITS>(
            wl2, x_thread, s2, b2, sum, remaining);
        result[3] += qdot_safe<U, values_per_thread, BITS>(
            wl3, x_thread, s3, b3, sum, remaining);
      } else {
        uchar4 zp_bytes = uchar4(
            zps[0], zps[zp_stride], zps[2 * zp_stride], zps[3 * zp_stride]);
        uchar4 zp_nibbles;
        if (BITS == 4) {
          const uint8_t shift = high_nibble ? 4u : 0u;
          zp_nibbles = (zp_bytes >> shift) & uchar4(0x0F);
        } else {
          zp_nibbles = zp_bytes;
        }
        result[0] += qdot_safe<U, values_per_thread, BITS>(
            wl0, x_thread, s0, -s0 * static_cast<U>(zp_nibbles.x), sum, remaining);
        result[1] += qdot_safe<U, values_per_thread, BITS>(
            wl1, x_thread, s1, -s1 * static_cast<U>(zp_nibbles.y), sum, remaining);
        result[2] += qdot_safe<U, values_per_thread, BITS>(
            wl2, x_thread, s2, -s2 * static_cast<U>(zp_nibbles.z), sum, remaining);
        result[3] += qdot_safe<U, values_per_thread, BITS>(
            wl3, x_thread, s3, -s3 * static_cast<U>(zp_nibbles.w), sum, remaining);
      }
    }
  }

  for (uint row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
  }

  if (k_split > 1) {
    if (simd_lane == 0) {
      for (uint row = 0; row < results_per_simdgroup; row++) {
        shared_results[simd_group * results_per_simdgroup + row] = result[row];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_slice == 0 && simd_lane == 0) {
      for (uint row = 0; row < results_per_simdgroup; row++) {
        U acc = 0;
        for (uint s = 0; s < k_split; s++) {
          acc += shared_results
              [(row_group * k_split + s) * results_per_simdgroup + row];
        }
        result[row] = acc;
      }
    }
  }

  const bool writer = (k_split == 1) || (k_slice == 0);

  if (writer && simd_lane == 0) {
    for (uint row = 0; row < results_per_simdgroup; row++) {
      U value = result[row];
      if (is_scale) {
        value = static_cast<U>(ab_scale) * value;
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
            static_cast<T>(shared_results[simd_lane]),
            hadamard_factors[global_out_idx]
        );
      }
    }
  } else {
    if (writer && simd_lane == 0) {
      for (uint row = 0; row < results_per_simdgroup; row++) {
        if (out_row + row < out_vec_size) {
          output[row] = static_cast<T>(result[row]);
        }
      }
    }
  }
}
