#include "../../common/dsl.h"

using namespace metal;

// Sums the `split_k` partial outputs produced by a split-K GEMM into the final
// output, in a single dispatch (the partials live contiguously as [split_k,
// n_elements]). Non-PUBLIC: a Metal-only helper used directly by GemmKernel.
template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(GemmSplitKReduce)(
    const device T* partials,
    device T* output,
    const device T* bias OPTIONAL(apply_bias),
    const constant uint& n_elements,
    const constant uint& split_k,
    const constant uint& group_count,
    const constant uint& n_cols,
    const constant float& out_scale,
    const constant uint& accumulate,
    const bool apply_bias SPECIALIZE,
    const uint group GROUPS(group_count),
    const uint local_id THREADS(256)
) {
  // Vectorized by 4: one thread reduces a vec<T,4> (n_elements is a multiple of
  // 4 since N is a multiple of the block-N tile). Fuses the elementwise
  // epilogue in the main-GEMM finalize order — scale the sum, accumulate the
  // prior output (read-modify-write) when requested, then add per-column bias;
  // any cross-column RHT is applied as a separate post-pass.
  const uint n4 = n_elements >> 2;
  const uint idx = group * 256u + local_id;
  if (idx >= n4) {
    return;
  }
  device vec<T, 4>* out4 = reinterpret_cast<device vec<T, 4>*>(output);
  const device vec<T, 4>* p4 =
      reinterpret_cast<const device vec<T, 4>*>(partials);
  float4 acc = float4(0.0f);
  for (uint p = 0u; p < split_k; ++p) {
    acc += float4(p4[p * n4 + idx]);
  }
  acc *= out_scale;
  if (accumulate != 0u) {
    acc += float4(out4[idx]);
  }
  if (apply_bias) {
    // 4 consecutive outputs share a row (n_cols is a multiple of 4), so this is
    // a vec4 of biases.
    const uint col = (idx * 4u) % n_cols;
    acc += float4(*reinterpret_cast<const device vec<T, 4>*>(bias + col));
  }
  out4[idx] = vec<T, 4>(acc);
}
