#include <metal_stdlib>
#include "../common/dsl.h"
#include "nf4_qmv_core.h"

// NF4-ZP QMV: `constant` codebook lookup plus a 4-bit per-group zero-point
// index (packed two-per-byte, row-major) selecting an additive offset from
// `nf4_zp_lut`. Dequant: out = scale * Σ (codebook[nibble] + zp_off) · x.
// Mirrors nf4_qmv_constant.metal exactly, with an extra `zero_points` buffer.
template <typename T, uint GROUP_SIZE>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 64)
KERNEL(Nf4QmvZp)(
    const device uint32_t* weights,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* input,
    device T* output,
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    const uint batch_idx GROUPS(batch_size),
    const uint out_block_idx GROUPS(out_vec_size.div_ceil(32)),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(8)
) {
  constexpr uint BITS = 4;
  constexpr uint packs_per_thread = 2;
  constexpr uint num_simdgroups = 8;
  constexpr uint results_per_simdgroup = 4;
  constexpr uint pack_factor = 8;
  constexpr uint bytes_per_pack = 4;
  constexpr uint values_per_thread = pack_factor * packs_per_thread;
  constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
  constexpr uint scale_step_per_thread = GROUP_SIZE / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)weights;
  thread float x_thread[values_per_thread];
  thread float result[results_per_simdgroup] = {0};

  const uint in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const uint in_vec_size_g = in_vec_size / GROUP_SIZE;
  const uint zp_stride = (in_vec_size_g + 1u) / 2u;
  const uint out_row =
      out_block_idx * (num_simdgroups * results_per_simdgroup) +
      simd_group * results_per_simdgroup;
  ws += out_row * in_vec_size_w + simd_lane * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lane / scale_step_per_thread;

  input += batch_idx * in_vec_size + simd_lane * values_per_thread;
  output += batch_idx * out_vec_size + out_row;

  // Group index this lane's scale/zp slot corresponds to, advancing by
  // `block_size / GROUP_SIZE` per outer iteration.
  uint group_idx = simd_lane / scale_step_per_thread;

  for (uint k = 0; k < in_vec_size; k += block_size) {
    (void)load_vector<T, float, values_per_thread, BITS>(input, x_thread);

    auto wl0 = (const device uint8_t*)(ws);
    auto wl1 = (const device uint8_t*)(ws + in_vec_size_w);
    auto wl2 = (const device uint8_t*)(ws + 2 * in_vec_size_w);
    auto wl3 = (const device uint8_t*)(ws + 3 * in_vec_size_w);

    float s0 = float(scales[0]);
    float s1 = float(scales[in_vec_size_g]);
    float s2 = float(scales[2 * in_vec_size_g]);
    float s3 = float(scales[3 * in_vec_size_g]);

    float z0 =
        float(nf4_zp_lookup(zero_points, out_row + 0, zp_stride, group_idx));
    float z1 =
        float(nf4_zp_lookup(zero_points, out_row + 1, zp_stride, group_idx));
    float z2 =
        float(nf4_zp_lookup(zero_points, out_row + 2, zp_stride, group_idx));
    float z3 =
        float(nf4_zp_lookup(zero_points, out_row + 3, zp_stride, group_idx));

    result[0] += qdot_nf4_zp<values_per_thread>(wl0, x_thread, s0, z0);
    result[1] += qdot_nf4_zp<values_per_thread>(wl1, x_thread, s1, z1);
    result[2] += qdot_nf4_zp<values_per_thread>(wl2, x_thread, s2, z2);
    result[3] += qdot_nf4_zp<values_per_thread>(wl3, x_thread, s3, z3);

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / GROUP_SIZE;
    group_idx += block_size / GROUP_SIZE;
    input += block_size;
  }

  for (uint row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
  }

  if (simd_lane == 0) {
    for (uint row = 0; row < results_per_simdgroup; row++) {
      output[row] = static_cast<T>(result[row]);
    }
  }
  (void)shared_results;
}
