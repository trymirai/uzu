#include <metal_stdlib>
#include "../common/dsl.h"
#include "nf4_qmv_core.h"

// NF4 QMV using a byte-batched 256-entry threadgroup `half2` codebook LUT
// (the access pattern that makes the int4 `awq-lut256` path beat scalar).
// Same VARIANTS/signature shape as Nf4QmvConstant (T=bfloat, GROUP_SIZE=64).
// Difference: cooperatively build `nf4_byte_lut[256]` where
//   nf4_byte_lut[b] = half2(nf4_codebook[b & 0x0f], nf4_codebook[(b>>4)&0x0f])
// then each packed weight byte is one threadgroup `half2` load → both
// nibbles' codebook values. Per-group bf16 `scale` is applied on accumulate
// (NOT in the table), exactly as qdot_nf4_constant.
template <typename T, uint GROUP_SIZE>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 64)
KERNEL(Nf4QmvByte256)(
    const device uint32_t* weights,
    const device T* scales,
    const device T* input,
    device T* output,
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    threadgroup half2 nf4_byte_lut[256],
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
  const uint out_row =
      out_block_idx * (num_simdgroups * results_per_simdgroup) +
      simd_group * results_per_simdgroup;
  ws += out_row * in_vec_size_w + simd_lane * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lane / scale_step_per_thread;

  input += batch_idx * in_vec_size + simd_lane * values_per_thread;
  output += batch_idx * out_vec_size + out_row;

  // Cooperatively build the byte-batched 256-entry codebook LUT.
  const uint tid = simd_group * METAL_SIMD_SIZE + simd_lane;
  nf4_init_byte_lut(nf4_byte_lut, tid, num_simdgroups * METAL_SIMD_SIZE);
  threadgroup_barrier(mem_flags::mem_threadgroup);

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

    result[0] +=
        qdot_nf4_byte_lut<values_per_thread>(wl0, x_thread, nf4_byte_lut, s0);
    result[1] +=
        qdot_nf4_byte_lut<values_per_thread>(wl1, x_thread, nf4_byte_lut, s1);
    result[2] +=
        qdot_nf4_byte_lut<values_per_thread>(wl2, x_thread, nf4_byte_lut, s2);
    result[3] +=
        qdot_nf4_byte_lut<values_per_thread>(wl3, x_thread, nf4_byte_lut, s3);

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / GROUP_SIZE;
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
