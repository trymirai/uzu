#include "../../common/dsl.h"

using namespace metal;

// Sums split_k partials ([split_k, n_elements]) into the final output. Non-PUBLIC.
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
    const uint col = (idx * 4u) % n_cols;
    acc += float4(*reinterpret_cast<const device vec<T, 4>*>(bias + col));
  }
  out4[idx] = vec<T, 4>(acc);
}
