#include <metal_stdlib>
#include "../common/dsl.h"
#include "nf4_qmv_core.h"

// NF4 QMV using a simdgroup-LOCAL threadgroup codebook. Each of the 8
// simdgroups owns its own contiguous 16-entry copy of the NF4 codebook in
// threadgroup memory (8 * 16 = 128 entries = 256 bytes total). Initialization
// is cooperative within each simdgroup (16 lanes write 1 entry each), then
// ordered with `simdgroup_barrier(mem_threadgroup)` — which only synchronizes
// the 32 lanes of THIS simdgroup, not the entire threadgroup.
//
// Phase-1 perf probe showed `Nf4QmvTgNoBarrier` (race version, no barrier at
// all) closes most of the small-K threadgroup-count gap on M4. The hypothesis
// here is that `simdgroup_barrier` is cheaper than `threadgroup_barrier` while
// remaining a correct mechanism to order our codebook stores/loads.
template <typename T, uint GROUP_SIZE>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 64)
KERNEL(Nf4QmvTgSimdbar)(
    const device uint32_t* weights,
    const device T* scales,
    const device T* input,
    device T* output,
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    threadgroup half codebook_tg[8 * 16],
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
  constexpr uint codebook_per_sg = 16;

  // Per-simdgroup base pointer into the 8-copy TG codebook.
  threadgroup half* my_cb = codebook_tg + simd_group * codebook_per_sg;

  // Cooperative init WITHIN THIS SIMDGROUP. The lower 16 lanes each write one
  // codebook entry; the upper 16 lanes are idle for the init only. Mirrors the
  // 16-case switch in `nf4_init_tg_codebook`.
  if (simd_lane < codebook_per_sg) {
    half v;
    switch (simd_lane) {
    case 0:
      v = -1.0h;
      break;
    case 1:
      v = -0.6961928h;
      break;
    case 2:
      v = -0.5250730h;
      break;
    case 3:
      v = -0.39491748h;
      break;
    case 4:
      v = -0.28444138h;
      break;
    case 5:
      v = -0.18477343h;
      break;
    case 6:
      v = -0.09105003h;
      break;
    case 7:
      v = 0.0h;
      break;
    case 8:
      v = 0.07958029h;
      break;
    case 9:
      v = 0.16093750h;
      break;
    case 10:
      v = 0.24611230h;
      break;
    case 11:
      v = 0.33791524h;
      break;
    case 12:
      v = 0.44070983h;
      break;
    case 13:
      v = 0.56261432h;
      break;
    case 14:
      v = 0.72295684h;
      break;
    default:
      v = 1.0h;
      break;
    }
    my_cb[simd_lane] = v;
  }

  // SIMDGROUP barrier: orders this simdgroup's 32 lanes' codebook stores with
  // the subsequent qdot loads. Does NOT synchronize across simdgroups, which
  // is fine since each simdgroup reads only its OWN copy of the codebook.
  simdgroup_barrier(mem_flags::mem_threadgroup);

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

    result[0] += qdot_nf4_tg<values_per_thread>(wl0, x_thread, my_cb, s0);
    result[1] += qdot_nf4_tg<values_per_thread>(wl1, x_thread, my_cb, s1);
    result[2] += qdot_nf4_tg<values_per_thread>(wl2, x_thread, my_cb, s2);
    result[3] += qdot_nf4_tg<values_per_thread>(wl3, x_thread, my_cb, s3);

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
