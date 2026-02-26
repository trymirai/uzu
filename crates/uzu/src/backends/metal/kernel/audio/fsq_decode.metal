#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
void fsq_decode(
    device const int* tokens,
    device T* out,
    device const int* lengths,
    const constant int& num_groups,
    const constant int& seq_len,
    const constant int& codebook_dim,
    const constant int* num_levels,
    const uint3 gid
) {
  const uint t = gid.x;
  const uint g = gid.y;
  const uint b = gid.z;

  if (t >= (uint)seq_len || g >= (uint)num_groups) {
    return;
  }

  const int len_b = lengths ? lengths[b] : seq_len;
  if ((int)t >= len_b) {
    // Zero all D codes for this token when masked.
    for (int d = 0; d < codebook_dim; ++d) {
      const uint out_c = g * (uint)codebook_dim + (uint)d;
      const uint out_idx =
          (b * (uint)(num_groups * codebook_dim) + out_c) * (uint)seq_len + t;
      out[out_idx] = (T)0;
    }
    return;
  }

  const uint token_idx = (b * (uint)num_groups + g) * (uint)seq_len + t;
  const int token = tokens[token_idx];

  // Compute dim bases (small D, do it per-thread).
  int base = 1;
  for (int d = 0; d < codebook_dim; ++d) {
    const int levels = num_levels[d];
    const int scale = levels / 2; // integer division (matches PyTorch // 2)
    const int offset = scale;

    // Euclidean modulo to keep result in [0, levels) even if token is negative
    // (tokens are expected nonnegative, but keep this robust).
    const int div = token / base;
    int code_nonneg = div % levels;
    if (code_nonneg < 0) {
      code_nonneg += levels;
    }
    const float code = ((float)(code_nonneg - offset)) / (float)scale;

    const uint out_c = g * (uint)codebook_dim + (uint)d;
    const uint out_idx =
        (b * (uint)(num_groups * codebook_dim) + out_c) * (uint)seq_len + t;
    out[out_idx] = (T)code;

    base *= levels;
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioFsqDecode)(
    device const int* tokens,
    device T* out,
    device const int* lengths,
    const constant int& num_groups,
    const constant int& seq_len,
    const constant int& codebook_dim,
    const constant int* num_levels,
    const constant int& batch_size,
    uint t AXIS(seq_len, 32),
    uint g AXIS(num_groups, 1),
    uint b AXIS(batch_size, 1)
) {
  fsq_decode<T>(
      tokens,
      out,
      lengths,
      num_groups,
      seq_len,
      codebook_dim,
      num_levels,
      uint3(t, g, b)
  );
}
