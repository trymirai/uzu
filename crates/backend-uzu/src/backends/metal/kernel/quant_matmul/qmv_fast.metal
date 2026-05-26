#include <metal_stdlib>
#include "../common/dsl.h"
#include "../hadamard_transform/hadamard_transform.h"
#include "nf4_common.h"
#include "quant_matmul.h"

// NF4-graft per-weight dequant: identical byte/nibble stride to int4 `qdot`
// (reads `w[2*i]`, `w[2*i+1]`, 4 nibbles low->high), but the nibble is a
// 16-entry NF4 codebook index instead of a uniform int4 magnitude. Scale-only
// (NO zero_points / NO bias): out = scale * Σ codebook[nibble] · x. This is
// the exact same math as `qdot_nf4_constant`; only the kernel skeleton
// (tiling/occupancy/loop) is the production QmvFast one.
template <int values_per_thread>
inline float qdot_nf4_graft(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale
) {
  using U4 = vec<float, 4>;
  float accum = 0;
  const thread U4* x4 = (const thread U4*)x_thread;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    uint8_t b0 = w[2 * i];
    uint8_t b1 = w[2 * i + 1];
    half h0 = nf4_codebook[b0 & 0x0f];
    half h1 = nf4_codebook[(b0 >> 4) & 0x0f];
    half h2 = nf4_codebook[b1 & 0x0f];
    half h3 = nf4_codebook[(b1 >> 4) & 0x0f];
    U4 w_vec = U4(float(h0), float(h1), float(h2), float(h3));
    accum += dot(x4[i], w_vec);
  }
  return scale * accum;
}

template <uint BITS, int values_per_thread>
inline float qdot_qmv_fast_experiment(
    const device uint8_t* w,
    const thread float* x_thread,
    const threadgroup bfloat2* lut,
    bool use_lut,
    bool use_nf4,
    float scale,
    float bias,
    float sum
) {
  if (BITS == 4) {
    if (use_nf4 && use_lut) {
      // NF4-LUT graft: 256-entry byte-batched LUT populated with NF4
      // codebook values (see kernel LUT-init). Reuses int4 byte-LUT dequant
      // helper; bias is 0 on this path (use_zero_points=false), so
      // `+ sum * bias` collapses to 0.
      return qdot_q4_byte_lut_bfloat<values_per_thread>(
          w,
          x_thread,
          lut,
          scale,
          bias,
          sum
      );
    } else if (use_nf4) {
      // Scale-only NF4 codebook dequant; bias/sum (zero-point term) unused.
      return qdot_nf4_graft<values_per_thread>(w, x_thread, scale);
    } else if (use_lut) {
      return qdot_q4_byte_lut_bfloat<values_per_thread>(
          w,
          x_thread,
          lut,
          scale,
          bias,
          sum
      );
    } else {
      return qdot<float, values_per_thread, 4>(w, x_thread, scale, bias, sum);
    }
  } else {
    return qdot<float, values_per_thread, BITS>(w, x_thread, scale, bias, sum);
  }
}

template <typename T, uint GROUP_SIZE, uint BITS>
VARIANTS(T, float, half, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4, 8)
PUBLIC KERNEL(QuantizedMatmulQmvFast)(
    const device uint32_t* weights,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device T* input,
    device T* output,
    const device int32_t* hadamard_factors OPTIONAL(use_hadamard),
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    const bool use_zero_points SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    const bool use_lut SPECIALIZE,
    const bool use_nf4 SPECIALIZE,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    threadgroup bfloat2 q4_lut[256],
    const uint batch_idx GROUPS(batch_size),
    const uint out_block_idx GROUPS(out_vec_size.div_ceil(32)),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(8)
) {
  constexpr uint packs_per_thread = BITS == 2 ? 1 : 2;
  constexpr uint num_simdgroups = 8;
  constexpr uint results_per_simdgroup = 4;
  constexpr uint pack_factor = get_pack_factor<BITS, 32>();
  constexpr uint bytes_per_pack = get_bytes_per_pack<BITS, 32>();
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

  if (use_mlx_quant) {
    biases += out_row * in_vec_size_g + simd_lane / scale_step_per_thread;
  } else {
    if (BITS == 4) {
      zp_stride = (in_vec_size_g + 1) / 2;
      zps = zero_points + out_row * zp_stride;
      uint g_offset = simd_lane / scale_step_per_thread;
      zps += g_offset / 2;
      high_nibble = (g_offset & 1);
    } else {
      zp_stride = in_vec_size_g;
      zps = zero_points + out_row * zp_stride;
      zps += simd_lane / scale_step_per_thread;
    }
  }

  input += batch_idx * in_vec_size + simd_lane * values_per_thread;
  output += batch_idx * out_vec_size + out_row;

  if (use_lut) {
    const uint tid = simd_group * METAL_SIMD_SIZE + simd_lane;
    if (use_nf4) {
      // NF4-LUT graft: 256-entry byte-batched codebook
      //   q4_lut[b] = bfloat2(nf4_codebook[b & 0x0f],
      //   nf4_codebook[(b>>4)&0x0f])
      // tid spans 0..255 (8 simdgroups × 32 lanes) so the cooperative fill
      // is one entry per lane. bfloat2 storage so the inner-loop convert
      // lowers to a 16-bit left shift (no fp-convert intrinsic) on M4.
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
  }

  for (uint k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, BITS>(input, x_thread);

    {
      auto wl0 = (const device uint8_t*)(ws);
      auto wl1 = (const device uint8_t*)(ws + in_vec_size_w);
      auto wl2 = (const device uint8_t*)(ws + 2 * in_vec_size_w);
      auto wl3 = (const device uint8_t*)(ws + 3 * in_vec_size_w);

      U s0 = static_cast<U>(scales[0]);
      U s1 = static_cast<U>(scales[in_vec_size_g]);
      U s2 = static_cast<U>(scales[2 * in_vec_size_g]);
      U s3 = static_cast<U>(scales[3 * in_vec_size_g]);

      if (use_mlx_quant) {
        U b0 = static_cast<U>(biases[0]);
        U b1 = static_cast<U>(biases[in_vec_size_g]);
        U b2 = static_cast<U>(biases[2 * in_vec_size_g]);
        U b3 = static_cast<U>(biases[3 * in_vec_size_g]);
        result[0] += qdot_qmv_fast_experiment<BITS, values_per_thread>(
            wl0,
            x_thread,
            q4_lut,
            use_lut,
            use_nf4,
            s0,
            b0,
            sum
        );
        result[1] += qdot_qmv_fast_experiment<BITS, values_per_thread>(
            wl1,
            x_thread,
            q4_lut,
            use_lut,
            use_nf4,
            s1,
            b1,
            sum
        );
        result[2] += qdot_qmv_fast_experiment<BITS, values_per_thread>(
            wl2,
            x_thread,
            q4_lut,
            use_lut,
            use_nf4,
            s2,
            b2,
            sum
        );
        result[3] += qdot_qmv_fast_experiment<BITS, values_per_thread>(
            wl3,
            x_thread,
            q4_lut,
            use_lut,
            use_nf4,
            s3,
            b3,
            sum
        );
      } else {
        // NF4 graft is scale-only: skip the zero-point load entirely so it
        // works with `zero_points == nullptr` (no zp buffer bound). The
        // bias it would produce is unused on the use_nf4 path anyway.
        uchar4 zp_nibbles = uchar4(0);
        if (!use_nf4) {
          uchar4 zp_bytes = uchar4(
              zps[0],
              zps[zp_stride],
              zps[2 * zp_stride],
              zps[3 * zp_stride]
          );
          if (BITS == 4) {
            const uint8_t shift = high_nibble ? 4u : 0u;
            zp_nibbles = (zp_bytes >> shift) & uchar4(0x0F);
          } else {
            zp_nibbles = zp_bytes;
          }
        } else {
          // EXPERIMENTAL probe: force NF4 to do the SAME memory work as AWQ
          // (4 device loads + shift/mask) using `ws` (always bound). The
          // loaded values feed `zp_nibbles` so the compiler can't DCE.
          // OUTPUTS WILL BE NUMERICALLY WRONG — perf-only experiment.
          uchar4 forced = uchar4(
              ws[0],
              ws[in_vec_size_w],
              ws[2 * in_vec_size_w],
              ws[3 * in_vec_size_w]
          );
          zp_nibbles = (forced >> 4) & uchar4(0x0F);
        }
        result[0] += qdot_qmv_fast_experiment<BITS, values_per_thread>(
            wl0,
            x_thread,
            q4_lut,
            use_lut,
            use_nf4,
            s0,
            -s0 * static_cast<U>(zp_nibbles.x),
            sum
        );
        result[1] += qdot_qmv_fast_experiment<BITS, values_per_thread>(
            wl1,
            x_thread,
            q4_lut,
            use_lut,
            use_nf4,
            s1,
            -s1 * static_cast<U>(zp_nibbles.y),
            sum
        );
        result[2] += qdot_qmv_fast_experiment<BITS, values_per_thread>(
            wl2,
            x_thread,
            q4_lut,
            use_lut,
            use_nf4,
            s2,
            -s2 * static_cast<U>(zp_nibbles.z),
            sum
        );
        result[3] += qdot_qmv_fast_experiment<BITS, values_per_thread>(
            wl3,
            x_thread,
            q4_lut,
            use_lut,
            use_nf4,
            s3,
            -s3 * static_cast<U>(zp_nibbles.w),
            sum
        );
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / GROUP_SIZE;
    if (use_mlx_quant) {
      biases += block_size / GROUP_SIZE;
    } else {
      if (BITS == 4) {
        zps += (block_size / GROUP_SIZE) / 2;
      } else {
        zps += block_size / GROUP_SIZE;
      }
    }
    input += block_size;
  }

  for (uint row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
  }

  if (use_hadamard) {
    if (simd_lane == 0) {
      for (uint row = 0; row < results_per_simdgroup; row++) {
        shared_results[simd_group * results_per_simdgroup + row] = result[row];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
      uint global_out_idx = out_block_idx * 32 + simd_lane;
      if (global_out_idx < out_vec_size) {
        output[simd_lane] = simdgroup_random_hadamard_transform(
            static_cast<ushort>(simd_lane),
            T(shared_results[simd_lane]),
            hadamard_factors[global_out_idx]
        );
      }
    }
  } else {
    if (simd_lane == 0) {
      for (uint row = 0; row < results_per_simdgroup; row++) {
        output[row] = static_cast<T>(result[row]);
      }
    }
  }
}
