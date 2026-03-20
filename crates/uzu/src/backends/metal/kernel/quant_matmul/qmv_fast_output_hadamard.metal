#include <metal_stdlib>
#include "../common/dsl.h"
#include "quant_matmul.metal"

using namespace metal;

// Fused QmvFast + output Hadamard transform.
// 8 simdgroups (256 threads) produce 32 contiguous outputs per threadgroup.
// After dot-product accumulation, results are staged in threadgroup memory,
// Hadamard-transformed via simd_shuffle_xor, then written to device memory.
// Eliminates the separate output Hadamard dispatch entirely.
template <typename T, int GROUP_SIZE, int BITS>
VARIANTS(T, float, half, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4, 8)
PUBLIC KERNEL(QuantizedMatmulQmvFastOutputHadamard)(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device T* x,
    device T* y,
    const device T* output_factors,
    const constant int& k,
    const constant int& n,
    const constant int& m,
    const bool use_zero_points SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    const uint tgid_x GROUPS(m),
    const uint tgid_y GROUPS((n + 32 - 1) / 32),
    const uint tgid_z GROUPS(1),
    const uint tid_x THREADS(32),
    const uint tid_y THREADS(8)
) {
  constexpr int packs_per_thread = BITS == 2 ? 1 : 2;
  constexpr int num_simdgroups = 8;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<BITS, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<BITS, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * METAL_SIMD_SIZE;
  constexpr int scale_step_per_thread = GROUP_SIZE / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const uint simd_gid = tid_y;
  const uint simd_lid = tid_x;

  const int in_vec_size_w = k * bytes_per_pack / pack_factor;
  const int in_vec_size_g = k / GROUP_SIZE;
  const int out_row = tgid_y * (num_simdgroups * results_per_simdgroup) +
                      simd_gid * results_per_simdgroup;

  // Early exit for simdgroups beyond n, but still participate in barrier
  if (out_row >= n) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return;
  }

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  int zp_stride = 0;
  const device uint8_t* zps = nullptr;
  bool high_nibble = false;

  if (use_mlx_quant) {
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  } else {
    if (BITS == 4) {
      zp_stride = (in_vec_size_g + 1) / 2;
      zps = zero_points + out_row * zp_stride;
      int g_offset = simd_lid / scale_step_per_thread;
      zps += g_offset / 2;
      high_nibble = (g_offset & 1);
    } else {
      zp_stride = in_vec_size_g;
      zps = zero_points + out_row * zp_stride;
      zps += simd_lid / scale_step_per_thread;
    }
  }

  const device T* x_local = x + tgid_x * k + simd_lid * values_per_thread;
  device T* y_base = y + tgid_x * n + tgid_y * (num_simdgroups * results_per_simdgroup);

  for (int ki = 0; ki < k; ki += block_size) {
    U sum = load_vector<T, U, values_per_thread, BITS>(x_local, x_thread);

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
        result[0] += qdot<U, values_per_thread, BITS>(wl0, x_thread, s0, b0, sum);
        result[1] += qdot<U, values_per_thread, BITS>(wl1, x_thread, s1, b1, sum);
        result[2] += qdot<U, values_per_thread, BITS>(wl2, x_thread, s2, b2, sum);
        result[3] += qdot<U, values_per_thread, BITS>(wl3, x_thread, s3, b3, sum);
      } else {
        uint8_t zp_byte0 = zps[0];
        uint8_t zp_byte1 = zps[zp_stride];
        uint8_t zp_byte2 = zps[2 * zp_stride];
        uint8_t zp_byte3 = zps[3 * zp_stride];
        U zp0 = static_cast<U>((BITS == 4 && high_nibble) ? (zp_byte0 >> 4) : (zp_byte0 & 0x0F));
        U zp1 = static_cast<U>((BITS == 4 && high_nibble) ? (zp_byte1 >> 4) : (zp_byte1 & 0x0F));
        U zp2 = static_cast<U>((BITS == 4 && high_nibble) ? (zp_byte2 >> 4) : (zp_byte2 & 0x0F));
        U zp3 = static_cast<U>((BITS == 4 && high_nibble) ? (zp_byte3 >> 4) : (zp_byte3 & 0x0F));
        if (BITS == 8) {
          zp0 = static_cast<U>(zp_byte0); zp1 = static_cast<U>(zp_byte1);
          zp2 = static_cast<U>(zp_byte2); zp3 = static_cast<U>(zp_byte3);
        }
        result[0] += qdot_zero_point<U, values_per_thread, BITS>(wl0, x_thread, s0, zp0);
        result[1] += qdot_zero_point<U, values_per_thread, BITS>(wl1, x_thread, s1, zp1);
        result[2] += qdot_zero_point<U, values_per_thread, BITS>(wl2, x_thread, s2, zp2);
        result[3] += qdot_zero_point<U, values_per_thread, BITS>(wl3, x_thread, s3, zp3);
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / GROUP_SIZE;
    if (use_mlx_quant) {
      biases += block_size / GROUP_SIZE;
    } else {
      if (BITS == 4) { zps += (block_size / GROUP_SIZE) / 2; }
      else { zps += block_size / GROUP_SIZE; }
    }
    x_local += block_size;
  }

  // ── Reduce within each simdgroup ──────────────────────────────────
  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
  }

  // ── Stage results in threadgroup memory ───────────────────────────
  if (simd_lid == 0) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      shared_results[simd_gid * results_per_simdgroup + row] = result[row];
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ── Hadamard transform on the 32 staged results (simdgroup 0) ────
  if (simd_gid == 0) {
    int global_out_idx = tgid_y * (num_simdgroups * results_per_simdgroup) + simd_lid;

    float value = shared_results[simd_lid];

    if (global_out_idx < n) {
      value *= float(output_factors[global_out_idx]);

      for (uint stride = 1; stride < METAL_SIMD_SIZE; stride <<= 1) {
        float partner = simd_shuffle_xor(value, static_cast<ushort>(stride));
        value = (simd_lid & stride) ? (partner - value) : (partner + value);
      }

      constexpr float normalization_factor = 1.0f / 5.656854249f;
      y_base[simd_lid] = T(value * normalization_factor);
    }
  }
}
