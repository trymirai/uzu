#include <metal_stdlib>
#include "../common/dsl.h"
#include "nf4_qmv_core.h"

// NF4 QMV with vec4-padded replicated TG codebook. Layout:
//   threadgroup half lut[16][4]   // 16 entries x 4 replicas, total 128 B
// Each of the 16 NF4 values is replicated 4 times. For a given nibble `n`,
// the 4 replicas live at bank-pair { 2n, 2n+1 } (entries 0-1 share a bank).
// Across all nibbles 0..15, banks 0..31 are all covered.
//
// Each lane reads `lut[nibble][simd_lane & 3]`, so the same logical nibble
// is sourced from one of 4 replica byte-offsets per lane → 32 lanes querying
// the SAME nibble fan out across 2 banks (16 lanes/bank, all same value =
// broadcast). Divergent nibbles spread across all 32 banks.
//
// This differs from Nf4QmvTgReplicated (where copies are contiguous and
// lanes are partitioned by `simd_lane >> 3`): here replicas are INTERLEAVED
// with entries, and lane partition is by `simd_lane & 3`.
template <typename T, uint GROUP_SIZE>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 64)
KERNEL(Nf4QmvTgVec4)(
    const device uint32_t* weights,
    const device T* scales,
    const device T* input,
    device T* output,
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    threadgroup half codebook_vec4[64],
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

  // Cooperative init: 64 entries = 16 nibbles x 4 replicas, all replicas of
  // entry `i` hold the same nf4 codebook value.
  const uint tid = simd_group * METAL_SIMD_SIZE + simd_lane;
  const uint tgp_size = num_simdgroups * METAL_SIMD_SIZE;
  for (uint i = tid; i < 64u; i += tgp_size) {
    // i = entry*4 + replica  →  entry = i >> 2
    codebook_vec4[i] = nf4_codebook[(i >> 2) & 0x0fu];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Per-lane replica selector.
  const uint replica = simd_lane & 3u;

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

    result[0] += qdot_nf4_vec4<values_per_thread>(
        wl0,
        x_thread,
        codebook_vec4,
        replica,
        s0
    );
    result[1] += qdot_nf4_vec4<values_per_thread>(
        wl1,
        x_thread,
        codebook_vec4,
        replica,
        s1
    );
    result[2] += qdot_nf4_vec4<values_per_thread>(
        wl2,
        x_thread,
        codebook_vec4,
        replica,
        s2
    );
    result[3] += qdot_nf4_vec4<values_per_thread>(
        wl3,
        x_thread,
        codebook_vec4,
        replica,
        s3
    );

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
