#include <metal_stdlib>
#include "../common/dsl.h"
#include "../generated/quantization_method.h"
#include "quant_matmul.h"

using namespace uzu::quantization_method;

template <typename T, uint GROUP_SIZE, uint BITS>
VARIANTS(T, float, half, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4, 8)
PUBLIC KERNEL(QuantizedMatmulQmv)(
    const device uint32_t* weights,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(quant_method == QuantizationMethod::AWQ),
    const device T* biases OPTIONAL(quant_method == QuantizationMethod::MLX),
    const device T* input,
    device T* output,
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    const QuantizationMethod quant_method SPECIALIZE,
    const uint batch_idx GROUPS(batch_size),
    const uint out_block_idx GROUPS(out_vec_size.div_ceil(8)),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(2)
) {
  constexpr uint num_simdgroups = 2;
  constexpr uint results_per_simdgroup = 4;
  constexpr uint packs_per_thread = 1;
  constexpr uint pack_factor = get_pack_factor<BITS, 32>();
  constexpr uint bytes_per_pack = get_bytes_per_pack<BITS, 32>();

  constexpr uint values_per_thread = pack_factor * packs_per_thread;
  constexpr uint block_size = values_per_thread * 32;
  constexpr uint scale_step_per_thread = GROUP_SIZE / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)weights;
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const uint in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const uint in_vec_size_g = (in_vec_size + GROUP_SIZE - 1) / GROUP_SIZE;
  const device T* scales_base = scales;
  const uint out_row =
      out_block_idx * (num_simdgroups * results_per_simdgroup) +
      simd_group * results_per_simdgroup;
  const uint used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lane * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lane / scale_step_per_thread;

    const uint zp_stride =
        BITS == 4 ? ((in_vec_size_g + 1) / 2) : in_vec_size_g;
    const device uint8_t* zps_row_base = nullptr;
    const device T* biases_row_base = nullptr;
    if (quant_method == QuantizationMethod::MLX) {
      biases_row_base = biases + out_row * in_vec_size_g;
    } else {
      zps_row_base = zero_points + out_row * zp_stride;
    }

    input += batch_idx * in_vec_size + simd_lane * values_per_thread;
    output += batch_idx * out_vec_size + out_row;

    uint k = 0;
    for (; k + block_size <= in_vec_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, BITS>(input, x_thread);

      for (uint row = 0;
           out_row + row < out_vec_size && row < results_per_simdgroup;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const uint row_idx = out_row + row;
        const device T* sr = scales_base + row_idx * in_vec_size_g;

        uint g = (k + simd_lane * values_per_thread) / GROUP_SIZE;
        U s = static_cast<U>(sr[g]);
        if (quant_method == QuantizationMethod::MLX) {
          const device T* bl = biases_row_base + row * in_vec_size_g;
          U b = static_cast<U>(bl[g]);
          result[row] +=
              qdot<U, values_per_thread, BITS>(wl, x_thread, s, b, sum);
        } else {
          const device uint8_t* zl = zps_row_base + row * zp_stride;
          U zp;
          if (BITS == 4) {
            uint8_t zp_b = zl[g >> 1];
            zp = static_cast<U>((g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
          } else {
            zp = static_cast<U>(zl[g]);
          }
          result[row] +=
              qdot<U, values_per_thread, BITS>(wl, x_thread, s, -s * zp, sum);
        }
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / GROUP_SIZE;
      input += block_size;
    }
    const uint thread_offset = simd_lane * values_per_thread;
    const int remaining =
        (k + thread_offset < in_vec_size)
            ? min(in_vec_size - k - thread_offset, values_per_thread)
            : 0;
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, BITS>(
          input,
          x_thread,
          remaining
      );

      for (uint row = 0;
           out_row + row < out_vec_size && row < results_per_simdgroup;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const uint row_idx = out_row + row;
        const device T* sr = scales_base + row_idx * in_vec_size_g;

        uint g = (k + simd_lane * values_per_thread) / GROUP_SIZE;
        U s = static_cast<U>(sr[g]);
        if (quant_method == QuantizationMethod::MLX) {
          const device T* bl = biases_row_base + row * in_vec_size_g;
          U b = static_cast<U>(bl[g]);
          result[row] +=
              qdot<U, values_per_thread, BITS>(wl, x_thread, s, b, sum);
        } else {
          const device uint8_t* zl = zps_row_base + row * zp_stride;
          U zp;
          if (BITS == 4) {
            uint8_t zp_b = zl[g >> 1];
            zp = static_cast<U>((g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
          } else {
            zp = static_cast<U>(zl[g]);
          }
          result[row] +=
              qdot<U, values_per_thread, BITS>(wl, x_thread, s, -s * zp, sum);
        }
      }
    }

    for (uint row = 0;
         out_row + row < out_vec_size && row < results_per_simdgroup;
         row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lane == 0) {
        output[row] = static_cast<T>(result[row]);
      }
    }
  } else {
    ws += used_out_row * in_vec_size_w +
          simd_lane * packs_per_thread * bytes_per_pack;
    scales += used_out_row * in_vec_size_g + simd_lane / scale_step_per_thread;

    const uint zp_stride =
        BITS == 4 ? ((in_vec_size_g + 1) / 2) : in_vec_size_g;
    const device uint8_t* zps_row_base = nullptr;
    const device T* biases_row_base = nullptr;
    if (quant_method == QuantizationMethod::MLX) {
      biases_row_base = biases + used_out_row * in_vec_size_g;
    } else {
      zps_row_base = zero_points + used_out_row * zp_stride;
    }

    input += batch_idx * in_vec_size + simd_lane * values_per_thread;
    output += batch_idx * out_vec_size + used_out_row;

    uint k = 0;
    for (; k + block_size <= in_vec_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, BITS>(input, x_thread);

      for (uint row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const uint row_idx = used_out_row + row;
        const device T* sr = scales_base + row_idx * in_vec_size_g;

        uint g = (k + simd_lane * values_per_thread) / GROUP_SIZE;
        U s = static_cast<U>(sr[g]);
        if (quant_method == QuantizationMethod::MLX) {
          const device T* bl = biases_row_base + row * in_vec_size_g;
          U b = static_cast<U>(bl[g]);
          result[row] +=
              qdot<U, values_per_thread, BITS>(wl, x_thread, s, b, sum);
        } else {
          const device uint8_t* zl = zps_row_base + row * zp_stride;
          U zp;
          if (BITS == 4) {
            uint8_t zp_b = zl[g >> 1];
            zp = static_cast<U>((g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
          } else {
            zp = static_cast<U>(zl[g]);
          }
          result[row] +=
              qdot<U, values_per_thread, BITS>(wl, x_thread, s, -s * zp, sum);
        }
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / GROUP_SIZE;
      input += block_size;
    }
    const uint thread_offset = simd_lane * values_per_thread;
    const int remaining =
        (k + thread_offset < in_vec_size)
            ? min(in_vec_size - k - thread_offset, values_per_thread)
            : 0;

    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, BITS>(
          input,
          x_thread,
          remaining
      );

      for (uint row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const uint row_idx = used_out_row + row;
        const device T* sr = scales_base + row_idx * in_vec_size_g;

        uint g = (k + simd_lane * values_per_thread) / GROUP_SIZE;
        U s = static_cast<U>(sr[g]);
        if (quant_method == QuantizationMethod::MLX) {
          const device T* bl = biases_row_base + row * in_vec_size_g;
          U b = static_cast<U>(bl[g]);
          result[row] += qdot_safe<U, values_per_thread, BITS>(
              wl,
              x_thread,
              s,
              b,
              sum,
              remaining
          );
        } else {
          const device uint8_t* zl = zps_row_base + row * zp_stride;
          U zp;
          if (BITS == 4) {
            uint8_t zp_b = zl[g >> 1];
            zp = static_cast<U>((g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
          } else {
            zp = static_cast<U>(zl[g]);
          }
          result[row] +=
              qdot<U, values_per_thread, BITS>(wl, x_thread, s, -s * zp, sum);
        }
      }
    }

    for (uint row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lane == 0) {
        output[row] = static_cast<T>(result[row]);
      }
    }
  }
}
