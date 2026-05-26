#include <metal_stdlib>
#include "../common/dsl.h"
#include "nf4_common.h"
#include "quant_matmul.h"

// Function-constant-codegen isolation experiment.
//
// Hypothesis: the +91% NF4-vs-AWQ gap inside QuantizedMatmulQmvFast's
// byte-batched LUT256 path is driven by Apple's PSO compiler treating the
// SPECIALIZE function-constant gating differently when the LUT-init branch
// switches from "uint nibbles -> bfloat2" to "nf4_codebook -> bfloat2" —
// even though the surrounding K-loop and the qdot helper are identical.
//
// Each entry below pins the formerly-SPECIALIZE flags as compile-time
// `constexpr bool` so the dead branches (use_mlx_quant, use_hadamard, the
// non-selected use_nf4 LUT init) are fully eliminated at AIR generation,
// not at PSO build. The PSO compiler then sees a smaller, fully-DCE'd
// kernel — eliminating function-constant codegen as a perf variable.
//
// Both kernels call the SAME `qdot_q4_byte_lut_bfloat` helper as the
// production QmvFast `use_lut=true` path. The only differences are:
//   * LUT-init body (int4 nibbles vs NF4 codebook values)
//   * zero-point load (AWQ variant) vs scale-only (NF4 variant)
//   * bias term passed to qdot (-scale*zp vs 0)
// Hadamard / MLX-quant / scalar paths from QmvFast are dropped (constexpr
// false, statically dead) — this kernel is LUT-only.

// =========================================================================
// Shared body. `USE_NF4` chooses between AWQ-int4 (false) and NF4-graft
// (true). `T = bfloat` for now (mirrors current bench coverage).
// =========================================================================
template <typename T, uint GROUP_SIZE, bool USE_NF4>
inline void qmv_fast_template_body(
    const device uint32_t* weights,
    const device T* scales,
    const device uint8_t* zero_points, // ignored if USE_NF4
    const device T* input,
    device T* output,
    threadgroup bfloat2 q4_lut[256],
    uint in_vec_size,
    uint out_vec_size,
    uint batch_idx,
    uint out_block_idx,
    uint simd_lane,
    uint simd_group
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
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const uint in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const uint in_vec_size_g = in_vec_size / GROUP_SIZE;
  const uint out_row =
      out_block_idx * (num_simdgroups * results_per_simdgroup) +
      simd_group * results_per_simdgroup;
  ws += out_row * in_vec_size_w + simd_lane * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lane / scale_step_per_thread;

  uint zp_stride = 0;
  const device uint8_t* zps = nullptr;
  bool high_nibble = false;
  if (!USE_NF4) {
    zp_stride = (in_vec_size_g + 1) / 2;
    zps = zero_points + out_row * zp_stride;
    uint g_offset = simd_lane / scale_step_per_thread;
    zps += g_offset / 2;
    high_nibble = (g_offset & 1);
  }

  input += batch_idx * in_vec_size + simd_lane * values_per_thread;
  output += batch_idx * out_vec_size + out_row;

  // LUT init (cooperative, 256 entries spread across 8x32 = 256 threads).
  const uint tid = simd_group * METAL_SIMD_SIZE + simd_lane;
  if (USE_NF4) {
    q4_lut[tid] = bfloat2(
        static_cast<bfloat>(nf4_codebook[tid & 0x0fu]),
        static_cast<bfloat>(nf4_codebook[(tid >> 4) & 0x0fu])
    );
  } else {
    q4_lut[tid] = bfloat2(
        static_cast<bfloat>(tid & 0x0f),
        static_cast<bfloat>((tid >> 4) & 0x0f)
    );
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, BITS>(input, x_thread);

    auto wl0 = (const device uint8_t*)(ws);
    auto wl1 = (const device uint8_t*)(ws + in_vec_size_w);
    auto wl2 = (const device uint8_t*)(ws + 2 * in_vec_size_w);
    auto wl3 = (const device uint8_t*)(ws + 3 * in_vec_size_w);

    U s0 = static_cast<U>(scales[0]);
    U s1 = static_cast<U>(scales[in_vec_size_g]);
    U s2 = static_cast<U>(scales[2 * in_vec_size_g]);
    U s3 = static_cast<U>(scales[3 * in_vec_size_g]);

    U b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    if (!USE_NF4) {
      uchar4 zp_bytes = uchar4(
          zps[0],
          zps[zp_stride],
          zps[2 * zp_stride],
          zps[3 * zp_stride]
      );
      const uint8_t shift = high_nibble ? 4u : 0u;
      uchar4 zp_nibbles = (zp_bytes >> shift) & uchar4(0x0F);
      b0 = -s0 * static_cast<U>(zp_nibbles.x);
      b1 = -s1 * static_cast<U>(zp_nibbles.y);
      b2 = -s2 * static_cast<U>(zp_nibbles.z);
      b3 = -s3 * static_cast<U>(zp_nibbles.w);
    }

    result[0] += qdot_q4_byte_lut_bfloat<values_per_thread>(
        wl0,
        x_thread,
        q4_lut,
        s0,
        b0,
        sum
    );
    result[1] += qdot_q4_byte_lut_bfloat<values_per_thread>(
        wl1,
        x_thread,
        q4_lut,
        s1,
        b1,
        sum
    );
    result[2] += qdot_q4_byte_lut_bfloat<values_per_thread>(
        wl2,
        x_thread,
        q4_lut,
        s2,
        b2,
        sum
    );
    result[3] += qdot_q4_byte_lut_bfloat<values_per_thread>(
        wl3,
        x_thread,
        q4_lut,
        s3,
        b3,
        sum
    );

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / GROUP_SIZE;
    if (!USE_NF4) {
      zps += (block_size / GROUP_SIZE) / 2;
    }
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
}

// =========================================================================
// AWQ-LUT variant: int4 weights + 4-bit packed zero-points + bf16 scale.
// Sanity-check kernel; should match the SPECIALIZE-based QmvFast within
// ±2pp at M=4 (same instruction stream).
// =========================================================================
template <typename T, uint GROUP_SIZE>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 64)
KERNEL(QmvFastTemplateAwqLut)(
    const device uint32_t* weights,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* input,
    device T* output,
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    threadgroup bfloat2 q4_lut[256],
    const uint batch_idx GROUPS(batch_size),
    const uint out_block_idx GROUPS(out_vec_size.div_ceil(32)),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(8)
) {
  qmv_fast_template_body<T, GROUP_SIZE, /*USE_NF4=*/false>(
      weights,
      scales,
      zero_points,
      input,
      output,
      q4_lut,
      in_vec_size,
      out_vec_size,
      batch_idx,
      out_block_idx,
      simd_lane,
      simd_group
  );
  (void)shared_results;
}

// =========================================================================
// NF4-LUT variant: NF4 codebook + bf16 scale (no zero-points).
// This is the deciding kernel: does removing SPECIALIZE gating close the
// 91% gap vs QuantizedMatmulQmvFast `use_nf4=true, use_lut=true`?
// =========================================================================
template <typename T, uint GROUP_SIZE>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 64)
KERNEL(QmvFastTemplateNf4Lut)(
    const device uint32_t* weights,
    const device T* scales,
    const device T* input,
    device T* output,
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    threadgroup bfloat2 q4_lut[256],
    const uint batch_idx GROUPS(batch_size),
    const uint out_block_idx GROUPS(out_vec_size.div_ceil(32)),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(8)
) {
  qmv_fast_template_body<T, GROUP_SIZE, /*USE_NF4=*/true>(
      weights,
      scales,
      /*zero_points=*/nullptr,
      input,
      output,
      q4_lut,
      in_vec_size,
      out_vec_size,
      batch_idx,
      out_block_idx,
      simd_lane,
      simd_group
  );
  (void)shared_results;
}
