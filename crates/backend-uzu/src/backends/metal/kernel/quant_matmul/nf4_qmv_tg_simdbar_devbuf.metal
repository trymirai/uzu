#include <metal_stdlib>
#include "../common/dsl.h"
#include "nf4_qmv_core.h"

// Production-flexible NF4 QMV: same simdgroup-LOCAL threadgroup codebook
// layout as `Nf4QmvTgSimdbar`, but the 16-entry codebook is sourced from a
// `const device half*` buffer set by the CPU at dispatch time (vs the
// constant `nf4_codebook[16]` baked into `nf4_common.h`).
//
// Motivation: production deployment of arbitrary 4-bit codebooks
// (Lloyd-Max, custom-trained, NF4 / other bit widths, etc.) requires the
// codebook to NOT be a compile-time constant. This kernel proves the
// simdbar fast path survives that flexibility.
//
// Layout (identical to `nf4_qmv_tg_simdbar.metal`):
//   threadgroup half codebook_tg[8 * 16] // 256 B, 8 per-simdgroup copies
//   threadgroup half* my_cb = codebook_tg + simd_group * 16;
//   simdgroup_barrier(mem_threadgroup); // orders this simdgroup's writes
//
// The ONLY difference vs the constant-codebook kernel: the cooperative
// init reads its 16 values from device memory (`codebook_dev[simd_lane]`)
// instead of switching over compile-time literals.
//
// NOTE: this kernel pulls in `qdot_nf4_tg` from `nf4_qmv_core.h`, which
// transitively includes `nf4_common.h`. That header's `nf4_codebook[16]`
// is NEVER referenced from this translation unit — the only codebook
// values that flow into the K-loop come from `codebook_dev`. The compiler
// has no way to constant-fold the device-buffer values into the dequant
// path, so this is a faithful simulation of the production scenario where
// the codebook is unknown at PSO compile time.
template <typename T, uint GROUP_SIZE>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 64)
KERNEL(Nf4QmvTgSimdbarDevbuf)(
    const device uint32_t* weights,
    const device T* scales,
    const device half* codebook_dev, // 16 entries, CPU-provided per dispatch
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

  // Cooperative init WITHIN THIS SIMDGROUP. The lower 16 lanes each load one
  // codebook entry from the device buffer and store it into this simdgroup's
  // private TG copy. The upper 16 lanes are idle for init only.
  if (simd_lane < codebook_per_sg) {
    my_cb[simd_lane] = codebook_dev[simd_lane];
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
