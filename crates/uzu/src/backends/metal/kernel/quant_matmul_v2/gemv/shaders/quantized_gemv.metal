#include <metal_stdlib>
#include "../../../definitions.metal"
#include "../../common/dequantize.h"

using namespace uzu::quantized_matmul;

template <typename T, int group_size, int bits, bool UseMlxQuant>
void quantized_gemv_dispatch(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* weight_source = (const device uint8_t*)w;
  typedef float U;
  thread U input_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int input_vector_size_packed = K * bytes_per_pack / pack_factor;
  const int input_vector_groups =
      (K + group_size - 1) / group_size;
  const device T* scales_base = scales;
  const int output_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                      simd_gid * results_per_simdgroup;
  const int clamped_output_row = min(N - results_per_simdgroup, output_row);

  if (output_row >= N) {
    return;
  }

  if (N < (num_simdgroups * results_per_simdgroup)) {
    weight_source +=
        output_row * input_vector_size_packed + simd_lid * packs_per_thread * bytes_per_pack;
    scales += output_row * input_vector_groups + simd_lid / scale_step_per_thread;

    const int zero_point_stride = bits == 4 ? ((input_vector_groups + 1) / 2) : input_vector_groups;
    const device uint8_t* zero_points_row_base = nullptr;
    const device T* biases_row_base = nullptr;
    if (UseMlxQuant) {
      biases_row_base = biases + output_row * input_vector_groups;
    } else {
      zero_points_row_base = zero_points + output_row * zero_point_stride;
    }

    x += tid.x * K + simd_lid * values_per_thread;
    y += tid.x * N + output_row;

    int k = 0;
    for (; k < K - block_size; k += block_size) {
      U sum = load_input_vector<T, U, values_per_thread, bits>(x, input_thread);

      for (int row = 0; output_row + row < N; row++) {
        if (row >= results_per_simdgroup)
          break;
        auto wl = (const device uint8_t*)(weight_source + row * input_vector_size_packed);
        const int row_idx = output_row + row;
        const device T* sr = scales_base + row_idx * input_vector_groups;

        int g = (k + simd_lid * values_per_thread) / group_size;
        U s = static_cast<U>(sr[g]);
        if (UseMlxQuant) {
          const device T* bl = biases_row_base + row * input_vector_groups;
          U b = static_cast<U>(bl[g]);
          result[row] +=
              quantized_dot_product<U, values_per_thread, bits>(wl, input_thread, s, b, sum);
        } else {
          const device uint8_t* zl = zero_points_row_base + row * zero_point_stride;
          U zp;
          if (bits == 4) {
            uint8_t zp_b = zl[g >> 1];
            zp = static_cast<U>((g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
          } else {
            zp = static_cast<U>(zl[g]);
          }
          result[row] +=
              quantized_dot_product_zero_point<U, values_per_thread, bits>(wl, input_thread, s, zp);
        }
      }

      weight_source += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(K - k - simd_lid * values_per_thread),
        0,
        values_per_thread
    );
    if (remaining > 0) {
      U sum = load_input_vector_checked<T, U, values_per_thread, bits>(
          x,
          input_thread,
          remaining
      );

      for (int row = 0; output_row + row < N; row++) {
        if (row >= results_per_simdgroup)
          break;
        auto wl = (const device uint8_t*)(weight_source + row * input_vector_size_packed);
        const int row_idx = output_row + row;
        const device T* sr = scales_base + row_idx * input_vector_groups;

        int g = (k + simd_lid * values_per_thread) / group_size;
        U s = static_cast<U>(sr[g]);
        if (UseMlxQuant) {
          const device T* bl = biases_row_base + row * input_vector_groups;
          U b = static_cast<U>(bl[g]);
          result[row] +=
              quantized_dot_product<U, values_per_thread, bits>(wl, input_thread, s, b, sum);
        } else {
          const device uint8_t* zl = zero_points_row_base + row * zero_point_stride;
          U zp;
          if (bits == 4) {
            uint8_t zp_b = zl[g >> 1];
            zp = static_cast<U>((g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
          } else {
            zp = static_cast<U>(zl[g]);
          }
          result[row] +=
              quantized_dot_product_zero_point<U, values_per_thread, bits>(wl, input_thread, s, zp);
        }
      }
    }

    for (int row = 0; output_row + row < N; row++) {
      if (row >= results_per_simdgroup)
        break;
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  } else {
    weight_source += clamped_output_row * input_vector_size_packed +
          simd_lid * packs_per_thread * bytes_per_pack;
    scales += clamped_output_row * input_vector_groups + simd_lid / scale_step_per_thread;

    const int zero_point_stride = bits == 4 ? ((input_vector_groups + 1) / 2) : input_vector_groups;
    const device uint8_t* zero_points_row_base = nullptr;
    const device T* biases_row_base = nullptr;
    if (UseMlxQuant) {
      biases_row_base = biases + clamped_output_row * input_vector_groups;
    } else {
      zero_points_row_base = zero_points + clamped_output_row * zero_point_stride;
    }

    x += tid.x * K + simd_lid * values_per_thread;
    y += tid.x * N + clamped_output_row;

    int k = 0;
    for (; k < K - block_size; k += block_size) {
      U sum = load_input_vector<T, U, values_per_thread, bits>(x, input_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(weight_source + row * input_vector_size_packed);
        const int row_idx = clamped_output_row + row;
        const device T* sr = scales_base + row_idx * input_vector_groups;

        int g = (k + simd_lid * values_per_thread) / group_size;
        U s = static_cast<U>(sr[g]);
        if (UseMlxQuant) {
          const device T* bl = biases_row_base + row * input_vector_groups;
          U b = static_cast<U>(bl[g]);
          result[row] +=
              quantized_dot_product<U, values_per_thread, bits>(wl, input_thread, s, b, sum);
        } else {
          const device uint8_t* zl = zero_points_row_base + row * zero_point_stride;
          U zp;
          if (bits == 4) {
            uint8_t zp_b = zl[g >> 1];
            zp = static_cast<U>((g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
          } else {
            zp = static_cast<U>(zl[g]);
          }
          result[row] +=
              quantized_dot_product_zero_point<U, values_per_thread, bits>(wl, input_thread, s, zp);
        }
      }

      weight_source += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(K - k - simd_lid * values_per_thread),
        0,
        values_per_thread
    );

    if (remaining > 0) {
      U sum = load_input_vector_checked<T, U, values_per_thread, bits>(
          x,
          input_thread,
          remaining
      );

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(weight_source + row * input_vector_size_packed);
        const int row_idx = clamped_output_row + row;
        const device T* sr = scales_base + row_idx * input_vector_groups;

        int g = (k + simd_lid * values_per_thread) / group_size;
        U s = static_cast<U>(sr[g]);
        if (UseMlxQuant) {
          const device T* bl = biases_row_base + row * input_vector_groups;
          U b = static_cast<U>(bl[g]);
          result[row] += quantized_dot_product_checked<U, values_per_thread, bits>(
              wl,
              input_thread,
              s,
              b,
              sum,
              remaining
          );
        } else {
          const device uint8_t* zl = zero_points_row_base + row * zero_point_stride;
          U zp;
          if (bits == 4) {
            uint8_t zp_b = zl[g >> 1];
            zp = static_cast<U>((g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
          } else {
            zp = static_cast<U>(zl[g]);
          }
          result[row] +=
              quantized_dot_product_zero_point<U, values_per_thread, bits>(wl, input_thread, s, zp);
        }
      }
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}

template <typename T, int group_size, int bits, bool use_mlx_quant>
void quantized_gemv_implementation(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  if (use_mlx_quant) {
    quantized_gemv_dispatch<T, group_size, bits, true>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        K,
        N,
        tid,
        simd_gid,
        simd_lid
    );
  } else {
    quantized_gemv_dispatch<T, group_size, bits, false>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        K,
        N,
        tid,
        simd_gid,
        simd_lid
    );
  }
}

template <typename T, int GROUP_SIZE, int BITS>
VARIANTS(T, float, half, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4, 8)
KERNEL(QuantizedMatmulGemvV2)(
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
    quantized_gemv_implementation<T, GROUP_SIZE, BITS, true>(
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
    quantized_gemv_implementation<T, GROUP_SIZE, BITS, false>(
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
