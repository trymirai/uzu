#include <metal_stdlib>
#include "../../common/dsl.h"
#include "../common/qdot.h"
#include "../gemm/common/quant_pack.h"

using namespace uzu::gemm;

// Lloyd-Max QMV: dequant = (codebook[weight_code] - bias_codebook[bias_code]) *
// scale.
//
// Storage (matches lalamo's lloyd_max format with bias_bits=4):
//   weights        : bit-packed weight codes (BITS-wide, lo bits first)
//   scales         : one T scale per group
//   codebook       : (1 << BITS) halves
//   bias_indices   : 4-bit packed bias codes (one per group, lo nibble = even)
//   bias_codebook  : BIAS_CODEBOOK_SIZE halves (bias_bits = 4)

#define NUM_SIMDGROUPS 8
#define BIAS_CODEBOOK_SIZE 16

template <typename T, uint GROUP_SIZE, uint BITS>
VARIANTS(T, float, half, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4)
PUBLIC KERNEL(QuantizedMatmulQmvLloydMax)(
    const device uint32_t* weights,
    const device T* scales,
    const device half* codebook,
    const device uint8_t* bias_indices,
    const device half* bias_codebook,
    const device T* input,
    device T* output,
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    threadgroup half codebook_tg[NUM_SIMDGROUPS][1 << BITS],
    threadgroup half bias_codebook_tg[NUM_SIMDGROUPS][BIAS_CODEBOOK_SIZE],
    const uint batch_idx GROUPS(batch_size),
    const uint out_block_idx GROUPS(out_vec_size.div_ceil(32)),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(NUM_SIMDGROUPS)
) {
  constexpr uint packs_per_thread = 2;
  constexpr uint results_per_simdgroup = 4;
  constexpr uint pack_factor = get_pack_factor<BITS, 32>();
  constexpr uint bytes_per_pack = get_bytes_per_pack<BITS, 32>();
  constexpr uint values_per_thread = pack_factor * packs_per_thread;
  constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
  constexpr uint scale_step_per_thread = GROUP_SIZE / values_per_thread;
  constexpr uint codebook_size = 1u << BITS;

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const device uint8_t* ws = (const device uint8_t*)weights;
  const uint in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const uint in_vec_size_g = in_vec_size / GROUP_SIZE;
  const uint bias_stride = (in_vec_size_g + 1) / 2;

  const uint out_row =
      out_block_idx * (NUM_SIMDGROUPS * results_per_simdgroup) +
      simd_group * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lane * packs_per_thread * bytes_per_pack;
  const uint scale_g_offset =
      out_row * in_vec_size_g + simd_lane / scale_step_per_thread;
  scales += scale_g_offset;

  uint bias_g_offset = simd_lane / scale_step_per_thread;
  const device uint8_t* bidx =
      bias_indices + out_row * bias_stride + bias_g_offset / 2;
  bool bias_high_nibble = (bias_g_offset & 1);
  static_assert(
      (block_size / GROUP_SIZE) % 2 == 0,
      "bias_high_nibble parity must be preserved across blocks"
  );

  input += batch_idx * in_vec_size + simd_lane * values_per_thread;
  output += batch_idx * out_vec_size + out_row;

  threadgroup half* codebook_sg = codebook_tg[simd_group];
  threadgroup half* bias_codebook_sg = bias_codebook_tg[simd_group];
  for (uint entry = simd_lane; entry < codebook_size;
       entry += METAL_SIMD_SIZE) {
    codebook_sg[entry] = codebook[entry];
  }
  for (uint entry = simd_lane; entry < BIAS_CODEBOOK_SIZE;
       entry += METAL_SIMD_SIZE) {
    bias_codebook_sg[entry] = bias_codebook[entry];
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  for (uint k = 0; k < in_vec_size; k += block_size) {
    load_vector_unscaled<T, U, values_per_thread>(input, x_thread);

    auto wl0 = (const device uint8_t*)(ws);
    auto wl1 = (const device uint8_t*)(ws + in_vec_size_w);
    auto wl2 = (const device uint8_t*)(ws + 2 * in_vec_size_w);
    auto wl3 = (const device uint8_t*)(ws + 3 * in_vec_size_w);

    U s0 = static_cast<U>(scales[0]);
    U s1 = static_cast<U>(scales[in_vec_size_g]);
    U s2 = static_cast<U>(scales[2 * in_vec_size_g]);
    U s3 = static_cast<U>(scales[3 * in_vec_size_g]);

    uchar4 bias_bytes = uchar4(
        bidx[0],
        bidx[bias_stride],
        bidx[2 * bias_stride],
        bidx[3 * bias_stride]
    );
    const uint8_t shift = bias_high_nibble ? 4u : 0u;
    uchar4 bias_codes = (bias_bytes >> shift) & uchar4(0x0F);

    U b0 = static_cast<U>(bias_codebook_sg[bias_codes.x]);
    U b1 = static_cast<U>(bias_codebook_sg[bias_codes.y]);
    U b2 = static_cast<U>(bias_codebook_sg[bias_codes.z]);
    U b3 = static_cast<U>(bias_codebook_sg[bias_codes.w]);

    result[0] += qdot_lloyd_max<U, values_per_thread, BITS>(
        wl0,
        x_thread,
        codebook_sg,
        s0,
        b0
    );
    result[1] += qdot_lloyd_max<U, values_per_thread, BITS>(
        wl1,
        x_thread,
        codebook_sg,
        s1,
        b1
    );
    result[2] += qdot_lloyd_max<U, values_per_thread, BITS>(
        wl2,
        x_thread,
        codebook_sg,
        s2,
        b2
    );
    result[3] += qdot_lloyd_max<U, values_per_thread, BITS>(
        wl3,
        x_thread,
        codebook_sg,
        s3,
        b3
    );

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / GROUP_SIZE;
    bidx += (block_size / GROUP_SIZE) / 2;
    input += block_size;
  }

  for (uint row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
  }

  qmv_write_direct_results<T, U, results_per_simdgroup>(
      result,
      output,
      simd_lane
  );
}
