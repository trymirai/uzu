#include <metal_stdlib>
#include "../common/dsl.h"
#include "../hadamard_transform/hadamard_transform.h"
#include "quant_matmul.h"

template <typename T, uint GROUP_SIZE, uint BITS>
VARIANTS(T, float, half, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4, 8)
PUBLIC KERNEL(QuantizedMatmulQmvFast)(
    const device uint32_t* weights,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device T* input,
    device T* output,
    const device int32_t* hadamard_factors OPTIONAL(use_hadamard),
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    const bool use_zero_points SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    const uint batch_idx GROUPS(batch_size),
    const uint out_block_idx GROUPS(out_vec_size.div_ceil(32)),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(8)
) {
  constexpr uint packs_per_thread = BITS == 2 ? 1 : 2;
  constexpr uint num_simdgroups = 8;
  constexpr uint results_per_simdgroup = 4;
  constexpr uint pack_factor = get_pack_factor<BITS, 32>();
  constexpr uint bytes_per_pack = get_bytes_per_pack<BITS, 32>();
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

  if (use_mlx_quant) {
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

  for (uint k = 0; k < in_vec_size; k += block_size) {
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

      if (use_mlx_quant) {
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
            wl0, x_thread, s0, -s0 * static_cast<U>(zp_nibbles.x), sum
        );
        result[1] += qdot<U, values_per_thread, BITS>(
            wl1, x_thread, s1, -s1 * static_cast<U>(zp_nibbles.y), sum
        );
        result[2] += qdot<U, values_per_thread, BITS>(
            wl2, x_thread, s2, -s2 * static_cast<U>(zp_nibbles.z), sum
        );
        result[3] += qdot<U, values_per_thread, BITS>(
            wl3, x_thread, s3, -s3 * static_cast<U>(zp_nibbles.w), sum
        );
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / GROUP_SIZE;
    if (use_mlx_quant) {
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
        output[simd_lane] = simdgroup_random_hadamard_transform(
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
