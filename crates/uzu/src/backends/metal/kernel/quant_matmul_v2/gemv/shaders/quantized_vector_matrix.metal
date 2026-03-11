#include <metal_stdlib>
#include "../../../definitions.metal"
#include "../../common/dequantize.h"

using namespace uzu::quantized_matmul;

template <typename T, int group_size, int bits, bool use_zero_points>
void quantized_vector_matrix_core(
    const device uint32_t* weight_source,
    const device T* scales,
    const device T* biases,
    const device uint8_t* zero_points,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  constexpr int num_simdgroups = 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int tn = 32 / pack_factor;
  constexpr int block_size = 32;

  typedef float U;
  typedef struct {
    uint8_t wi[tn * bytes_per_pack];
  } vec_w;

  thread vec_w w_local;
  thread U result[tn * pack_factor] = {0};

  const int output_vector_size_packed = out_vec_size * bytes_per_pack / pack_factor;
  const int output_vector_groups = out_vec_size / group_size;
  const int output_column = pack_factor * tn * (tid.y * num_simdgroups + simd_gid);

  if (output_column >= out_vec_size) {
    return;
  }

  const int output_group = output_column / group_size;
  const int zero_point_row_stride =
      use_zero_points
          ? ((bits == 4) ? ((output_vector_groups + 1) / 2) : output_vector_groups)
          : 0;

  const device uint8_t* weight_source_pointer = (const device uint8_t*)weight_source +
                                 output_column * bytes_per_pack / pack_factor +
                                 simd_lid * output_vector_size_packed;
  const device T* scales_base = scales;
  const device T* biases_base = biases;
  x += tid.x * in_vec_size + simd_lid;
  y += tid.x * out_vec_size + output_column;

  for (int k_base = 0; k_base < in_vec_size; k_base += block_size) {
    const int k_index = k_base + simd_lid;
    const bool active = k_index < in_vec_size;

    U x_local = active ? static_cast<U>(*x) : U(0);
    U scale =
        active
            ? static_cast<U>(scales_base[k_index * output_vector_groups + output_group])
            : U(0);
    U bias;

    if (use_zero_points && active) {
      U zp;
      if (bits == 4) {
        const device uint8_t* row_ptr = zero_points + k_index * zero_point_row_stride;
        const device uint8_t* zp_ptr = row_ptr + (output_group >> 1);
        uint8_t zp_byte = *zp_ptr;
        bool high_nibble = (output_group & 1) != 0;
        zp = static_cast<U>(
            high_nibble ? ((zp_byte >> 4) & 0x0F) : (zp_byte & 0x0F)
        );
      } else {
        const device uint8_t* row_ptr = zero_points + k_index * zero_point_row_stride;
        const device uint8_t* zp_ptr = row_ptr + output_group;
        zp = static_cast<U>(*zp_ptr);
      }
      bias = -scale * zp;
    } else if (active) {
      bias = static_cast<U>(biases_base[k_index * output_vector_groups + output_group]);
    } else {
      bias = U(0);
    }

    if (active) {
      w_local = *((device vec_w*)weight_source_pointer);
    }

    quantized_outer_product<U, tn * pack_factor, bits>(
        (thread uint8_t*)&w_local,
        x_local,
        scale,
        bias,
        result
    );

    x += block_size;
    scales += block_size * output_vector_groups;
    if (!use_zero_points) {
      biases += block_size * output_vector_groups;
    }
    weight_source_pointer += block_size * output_vector_size_packed;
  }

#pragma clang loop unroll(full)
  for (int k = 0; k < tn * pack_factor; k++) {
    result[k] = simd_sum(result[k]);
  }

  if (simd_lid == 0) {
#pragma clang loop unroll(full)
    for (int k = 0; k < tn * pack_factor; k++) {
      y[k] = static_cast<T>(result[k]);
    }
  }
}

template <typename T, int group_size, int bits>
void quantized_vector_matrix_affine_bias(
    const device uint32_t* weight_source,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  quantized_vector_matrix_core<T, group_size, bits, false>(
      weight_source,
      scales,
      biases,
      nullptr,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid
  );
}

template <typename T, int group_size, int bits>
void quantized_vector_matrix_zero_point(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  quantized_vector_matrix_core<T, group_size, bits, true>(
      w,
      scales,
      nullptr,
      zero_points,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid
  );
}

template <typename T, int group_size, int bits, bool use_mlx_quant>
void quantized_vector_matrix_implementation(
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
    quantized_vector_matrix_affine_bias<T, group_size, bits>(
        w,
        scales,
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
    quantized_vector_matrix_zero_point<T, group_size, bits>(
        w,
        scales,
        zero_points,
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
KERNEL(QuantizedMatmulVectorMatrixV2)(
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
    const uint tgid_y GROUPS((n + 64 - 1) / 64),
    const uint tgid_z GROUPS(1),
    const uint tid_x THREADS(32),
    const uint tid_y THREADS(2)
) {
  const uint3 tid = uint3(tgid_x, tgid_y, tgid_z);
  const uint simd_gid = tid_y;
  const uint simd_lid = tid_x;

  if (use_mlx_quant) {
    quantized_vector_matrix_implementation<T, GROUP_SIZE, BITS, true>(
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
    quantized_vector_matrix_implementation<T, GROUP_SIZE, BITS, false>(
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
