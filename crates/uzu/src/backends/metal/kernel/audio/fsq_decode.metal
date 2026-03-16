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
    const constant int* dim_base_index,
    const uint3 gid
) {
  const uint t = gid.x;
  const uint gd = gid.y;
  const uint b = gid.z;
  const uint channels = (uint)(num_groups * codebook_dim);

  if (t >= (uint)seq_len || gd >= channels) {
    return;
  }

  const uint g = gd / (uint)codebook_dim;
  const uint d = gd % (uint)codebook_dim;

  const int len_b = lengths ? lengths[b] : seq_len;
  const uint out_idx = (b * channels + gd) * (uint)seq_len + t;
  if ((int)t >= len_b) {
    out[out_idx] = (T)0;
    return;
  }

  const uint token_idx = (b * (uint)num_groups + g) * (uint)seq_len + t;
  const int token = tokens[token_idx];

  const int levels = num_levels[d];
  const int scale = levels / 2; // integer division (matches PyTorch // 2)
  const int offset = scale;
  const int base = dim_base_index[d];

  // Euclidean modulo to keep result in [0, levels) even if token is negative
  // (tokens are expected nonnegative, but keep this robust).
  const int div = token / base;
  int code_nonneg = div % levels;
  if (code_nonneg < 0) {
    code_nonneg += levels;
  }
  const float code = ((float)(code_nonneg - offset)) / (float)scale;
  out[out_idx] = (T)code;
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(AudioFsqDecode)(
    device const int* tokens,
    device T* out,
    device const int* lengths,
    const constant int& num_groups,
    const constant int& seq_len,
    const constant int& codebook_dim,
    const constant int* num_levels,
    const constant int* dim_base_index,
    const constant int& batch_size,
    uint t AXIS(seq_len, 32),
    uint gd AXIS(num_groups * codebook_dim, 1),
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
      dim_base_index,
      uint3(t, gd, b)
  );
}
