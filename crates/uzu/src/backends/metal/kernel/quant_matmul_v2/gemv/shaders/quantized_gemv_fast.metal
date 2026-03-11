#include <metal_stdlib>
#include "../../../definitions.metal"
#include "../../common/dequantize.h"

using namespace uzu::quantized_matmul;

template <typename T, int group_size, int bits, bool use_mlx_quant>
void quantized_gemv_fast_implementation(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  const device uint8_t* weight_source = (const device uint8_t*)w;
  typedef float U;
  thread U input_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int input_vector_size_packed = in_vec_size * bytes_per_pack / pack_factor;
  const int input_vector_groups = in_vec_size / group_size;
  const int output_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                      simd_gid * results_per_simdgroup;
  weight_source += output_row * input_vector_size_packed + simd_lid * packs_per_thread * bytes_per_pack;
  scales += output_row * input_vector_groups + simd_lid / scale_step_per_thread;

  int zero_point_stride = 0;
  const device uint8_t* zero_points_pointer = nullptr;
  bool high_nibble = false;

  if (use_mlx_quant) {
    biases += output_row * input_vector_groups + simd_lid / scale_step_per_thread;
  } else {
    if (bits == 4) {
      zero_point_stride = (input_vector_groups + 1) / 2;
      zero_points_pointer = zero_points + output_row * zero_point_stride;
      int g_offset = simd_lid / scale_step_per_thread;
      zero_points_pointer += g_offset / 2;
      high_nibble = (g_offset & 1);
    } else {
      zero_point_stride = input_vector_groups;
      zero_points_pointer = zero_points + output_row * zero_point_stride;
      zero_points_pointer += simd_lid / scale_step_per_thread;
    }
  }

  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + output_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_input_vector<T, U, values_per_thread, bits>(x, input_thread);

    {
      auto wl0 = (const device uint8_t*)(weight_source);
      auto wl1 = (const device uint8_t*)(weight_source + input_vector_size_packed);
      auto wl2 = (const device uint8_t*)(weight_source + 2 * input_vector_size_packed);
      auto wl3 = (const device uint8_t*)(weight_source + 3 * input_vector_size_packed);

      U s0 = static_cast<U>(scales[0]);
      U s1 = static_cast<U>(scales[input_vector_groups]);
      U s2 = static_cast<U>(scales[2 * input_vector_groups]);
      U s3 = static_cast<U>(scales[3 * input_vector_groups]);

      if (use_mlx_quant) {
        U b0 = static_cast<U>(biases[0]);
        U b1 = static_cast<U>(biases[input_vector_groups]);
        U b2 = static_cast<U>(biases[2 * input_vector_groups]);
        U b3 = static_cast<U>(biases[3 * input_vector_groups]);
        result[0] +=
            quantized_dot_product<U, values_per_thread, bits>(wl0, input_thread, s0, b0, sum);
        result[1] +=
            quantized_dot_product<U, values_per_thread, bits>(wl1, input_thread, s1, b1, sum);
        result[2] +=
            quantized_dot_product<U, values_per_thread, bits>(wl2, input_thread, s2, b2, sum);
        result[3] +=
            quantized_dot_product<U, values_per_thread, bits>(wl3, input_thread, s3, b3, sum);
      } else {
        uint8_t zp_byte0 = zero_points_pointer[0];
        uint8_t zp_byte1 = zero_points_pointer[zero_point_stride];
        uint8_t zp_byte2 = zero_points_pointer[2 * zero_point_stride];
        uint8_t zp_byte3 = zero_points_pointer[3 * zero_point_stride];
        U zp0 = static_cast<U>(
            (bits == 4 && high_nibble) ? (zp_byte0 >> 4) : (zp_byte0 & 0x0F)
        );
        U zp1 = static_cast<U>(
            (bits == 4 && high_nibble) ? (zp_byte1 >> 4) : (zp_byte1 & 0x0F)
        );
        U zp2 = static_cast<U>(
            (bits == 4 && high_nibble) ? (zp_byte2 >> 4) : (zp_byte2 & 0x0F)
        );
        U zp3 = static_cast<U>(
            (bits == 4 && high_nibble) ? (zp_byte3 >> 4) : (zp_byte3 & 0x0F)
        );
        if (bits == 8) {
          zp0 = static_cast<U>(zp_byte0);
          zp1 = static_cast<U>(zp_byte1);
          zp2 = static_cast<U>(zp_byte2);
          zp3 = static_cast<U>(zp_byte3);
        }
        result[0] +=
            quantized_dot_product_zero_point<U, values_per_thread, bits>(wl0, input_thread, s0, zp0);
        result[1] +=
            quantized_dot_product_zero_point<U, values_per_thread, bits>(wl1, input_thread, s1, zp1);
        result[2] +=
            quantized_dot_product_zero_point<U, values_per_thread, bits>(wl2, input_thread, s2, zp2);
        result[3] +=
            quantized_dot_product_zero_point<U, values_per_thread, bits>(wl3, input_thread, s3, zp3);
      }
    }

    weight_source += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    if (use_mlx_quant) {
      biases += block_size / group_size;
    } else {
      if (bits == 4) {
        zero_points_pointer += (block_size / group_size) / 2;
      } else {
        zero_points_pointer += block_size / group_size;
      }
    }
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int GROUP_SIZE, int BITS>
VARIANTS(T, float, half, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4, 8)
KERNEL(QuantizedMatmulGemvFastV2)(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device T* x,
    device T* y,
    const constant int& k,
    const constant int& n,
    const constant int& m,
    const bool use_zero_points SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    const uint tgid_x GROUPS(m),
    const uint tgid_y GROUPS((n + 8 - 1) / 8),
    const uint tgid_z GROUPS(1),
    const uint tid_x THREADS(32),
    const uint tid_y THREADS(2)
) {
  const uint3 tid = uint3(tgid_x, tgid_y, tgid_z);
  const uint simd_gid = tid_y;
  const uint simd_lid = tid_x;

  if (use_mlx_quant) {
    quantized_gemv_fast_implementation<T, GROUP_SIZE, BITS, true>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        k,
        n,
        tid,
        simd_gid,
        simd_lid
    );
  } else {
    quantized_gemv_fast_implementation<T, GROUP_SIZE, BITS, false>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        k,
        n,
        tid,
        simd_gid,
        simd_lid
    );
  }
}
