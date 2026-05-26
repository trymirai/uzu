#include <metal_stdlib>
#include "../common/dsl.h"
#include "quant_matmul.h"

// Compile-time visibility experiment kernel.
//
// Hypothesis (motivating this whole file): the +91% NF4-vs-AWQ gap inside
// QuantizedMatmulQmvFast `use_lut=true && use_nf4=true` (nf4-lut-grft) is
// caused by the MSL/LLVM compiler having compile-time visibility into the
// `constant nf4_codebook[16]` array — even when the bench LUT is populated
// from it at runtime. Constant-folding / value-tracking through the
// half-codebook may emit different code than the int4-nibble LUT init.
//
// This kernel cuts that link cleanly: the 256-entry bfloat2 LUT is computed
// CPU-side (one entry per packed weight byte, both nibbles' codebook values)
// and bound as a device buffer. The kernel reads from device memory into
// threadgroup memory and the compiler cannot constant-fold device-memory
// values. If `qmv-fast-nf4-precomputed` matches `awq-lut256` (≈-15% vs
// scalar) then compile-time codebook visibility was the killer; if it stays
// near +91% then the bottleneck lives in the K-loop instruction stream.
//
// IMPORTANT: this kernel intentionally does NOT include `nf4_common.h` (the
// header that defines `constant nf4_codebook[16]`) so the compiler has no
// way to "see through" the precomputed table to the original 16 NF4 values.
//
// Structure: hardcoded to `use_nf4=true, use_lut=true, use_zero_points=false,
// use_mlx_quant=false, use_hadamard=false`. No SPECIALIZE function-constants
// — the whole point is to isolate the LUT-source variable from PSO codegen.
// The K-loop body matches the existing nf4-lut-grft path one-for-one
// (`qdot_q4_byte_lut_bfloat`, scale-only, bias=0).
template <typename T, uint GROUP_SIZE>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 64)
KERNEL(QmvFastNf4Precomputed)(
    const device uint32_t* weights,
    const device T* scales,
    const device bfloat2* precomputed_lut, // 256 entries, CPU-precomputed
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

  input += batch_idx * in_vec_size + simd_lane * values_per_thread;
  output += batch_idx * out_vec_size + out_row;

  // LUT init: flat device-to-threadgroup copy. NO reference to
  // `nf4_codebook[]` (which is intentionally not included). The compiler
  // cannot fold device-memory values, so the 16-entry NF4 codebook never
  // "appears" inside this PSO at any compile stage.
  //
  // tgp_size = num_simdgroups * METAL_SIMD_SIZE = 256, so every lane copies
  // exactly one entry — but written as a loop in case the threadgroup
  // geometry changes.
  const uint tid = simd_group * METAL_SIMD_SIZE + simd_lane;
  constexpr uint tgp_size = num_simdgroups * METAL_SIMD_SIZE;
  for (uint i = tid; i < 256u; i += tgp_size) {
    q4_lut[i] = precomputed_lut[i];
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

    // Scale-only NF4 path: bias=0 (no zero-points), so `+ sum * bias`
    // collapses inside the helper. Identical inner loop to the existing
    // nf4-lut-grft kernel path in qmv_fast.metal (use_nf4=true,use_lut=true).
    result[0] += qdot_q4_byte_lut_bfloat<values_per_thread>(
        wl0,
        x_thread,
        q4_lut,
        s0,
        /*bias=*/0.0f,
        sum
    );
    result[1] += qdot_q4_byte_lut_bfloat<values_per_thread>(
        wl1,
        x_thread,
        q4_lut,
        s1,
        /*bias=*/0.0f,
        sum
    );
    result[2] += qdot_q4_byte_lut_bfloat<values_per_thread>(
        wl2,
        x_thread,
        q4_lut,
        s2,
        /*bias=*/0.0f,
        sum
    );
    result[3] += qdot_q4_byte_lut_bfloat<values_per_thread>(
        wl3,
        x_thread,
        q4_lut,
        s3,
        /*bias=*/0.0f,
        sum
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
