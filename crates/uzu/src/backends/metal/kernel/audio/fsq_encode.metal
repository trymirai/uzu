#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

inline float round_ties_to_even(const float x) {
  const float f = floor(x);
  const float frac = x - f;
  if (frac < 0.5f) {
    return f;
  }
  if (frac > 0.5f) {
    return f + 1.0f;
  }
  const int fi = (int)f; // f is integral
  return ((fi & 1) != 0) ? (f + 1.0f) : f;
}

template <typename T>
void fsq_encode(
    device const T* input,     // [B, G*D, T]
    device int* tokens,        // [B, G, T]
    device const int* lengths, // [B]
    const constant int& num_groups,
    const constant int& seq_len,
    const constant int& codebook_dim,
    const constant int* num_levels,
    const constant int* dim_base_index,
    const constant float& eps,
    const uint3 gid
) {
  const uint t = gid.x;
  const uint g = gid.y;
  const uint b = gid.z;

  if (t >= (uint)seq_len || g >= (uint)num_groups) {
    return;
  }

  const int len_b = lengths ? lengths[b] : seq_len;
  const uint out_idx = (b * (uint)num_groups + g) * (uint)seq_len + t;
  if ((int)t >= len_b) {
    tokens[out_idx] = 0;
    return;
  }

  const uint in_base =
      (b * (uint)(num_groups * codebook_dim) + g * (uint)codebook_dim) *
          (uint)seq_len +
      t;

  int token = 0;
  for (int d = 0; d < codebook_dim; ++d) {
    const int levels = num_levels[d];
    const int scale_i = levels / 2;
    const float output_scale = ((float)(levels - 1)) * 0.5f * (1.0f - eps);
    const float output_offset = (levels % 2 == 0) ? 0.5f : 0.0f;
    const float input_shift = tan(output_offset / output_scale);

    const float x = float(input[in_base + (uint)d * (uint)seq_len]);
    const float compressed =
        output_scale * tanh(x + input_shift) - output_offset;
    const float rounded = round_ties_to_even(compressed);

    int code_nonneg = (int)rounded + scale_i;
    code_nonneg = clamp(code_nonneg, 0, levels - 1);
    token += code_nonneg * dim_base_index[d];
  }

  tokens[out_idx] = token;
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioFsqEncode)(
    device const T* input,
    device int* tokens,
    device const int* lengths,
    const constant int& num_groups,
    const constant int& seq_len,
    const constant int& codebook_dim,
    const constant int* num_levels,
    const constant int* dim_base_index,
    const constant float& eps,
    const constant int& batch_size,
    uint t AXIS(seq_len, 32),
    uint g AXIS(num_groups, 1),
    uint b AXIS(batch_size, 1)
) {
  fsq_encode<T>(
      input,
      tokens,
      lengths,
      num_groups,
      seq_len,
      codebook_dim,
      num_levels,
      dim_base_index,
      eps,
      uint3(t, g, b)
  );
}
