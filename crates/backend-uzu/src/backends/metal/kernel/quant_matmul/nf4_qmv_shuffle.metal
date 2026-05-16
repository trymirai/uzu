#include <metal_stdlib>
#include "../common/dsl.h"
#include "nf4_qmv_core.h"

// NF4 QMV using a ZERO-MEMORY register *shuffle* codebook.
//
// Lanes 0..(CODEBOOK_SIZE-1) of the 32-lane simdgroup each hold one codebook
// entry in a register (computed once via a switch-of-literals — NO array, so
// no spill, NO `constant`/`threadgroup` codebook). Per-weight dequant fetches
// the needed entry via `simd_shuffle(my_entry, nibble)`: a pure cross-lane
// register op with no memory/LSU traffic (vs `Nf4QmvConstant`'s constant-space
// gather and `Nf4QmvByte256`'s threadgroup LUT).
//
// CODEBOOK_SIZE ∈ {8,16,32}:
//   16 → the real NF4 16-value codebook (numerically equivalent to
//        `Nf4QmvConstant`).
//   8/32 → synthetic monotonic tables (timing probes only); the kernel is
//        still correct vs a CPU reference using the SAME table. Weights stay
//        4-bit nibbles; S=8 masks the nibble to 3 bits.
// Same VARIANTS/signature/scale handling as `Nf4QmvConstant` (T=bfloat,
// GROUP_SIZE=64, bf16 per-group scale, no zero_points/bias).
template <typename T, uint GROUP_SIZE, uint CODEBOOK_SIZE>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 64)
VARIANTS(CODEBOOK_SIZE, 8, 16, 32)
KERNEL(Nf4QmvShuffle)(
    const device uint32_t* weights,
    const device T* scales,
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

  // This lane's register-held codebook entry (zero memory: switch of
  // literals, no array). Computed ONCE before the K loop.
  const half my_entry = nf4_my_shuffle_entry_n<CODEBOOK_SIZE>(simd_lane);

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

    result[0] += qdot_nf4_shuffle<values_per_thread, CODEBOOK_SIZE>(
        wl0,
        x_thread,
        my_entry,
        s0
    );
    result[1] += qdot_nf4_shuffle<values_per_thread, CODEBOOK_SIZE>(
        wl1,
        x_thread,
        my_entry,
        s1
    );
    result[2] += qdot_nf4_shuffle<values_per_thread, CODEBOOK_SIZE>(
        wl2,
        x_thread,
        my_entry,
        s2
    );
    result[3] += qdot_nf4_shuffle<values_per_thread, CODEBOOK_SIZE>(
        wl3,
        x_thread,
        my_entry,
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
