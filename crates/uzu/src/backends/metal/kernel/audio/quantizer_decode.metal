#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

template <typename T>
void quantizer_decode(
    device const uint* tokens,          // [B, K, T]
    device const int* lengths,          // [B]
    device const T* semantic_codebook,  // [semantic_cardinality, codebook_dim]
    device const T* semantic_out_proj,  // [input_dim, codebook_dim]
    device const T* semantic_out_bias,  // [input_dim]
    device const T* residual_codebooks, // [R, residual_cardinality, codebook_dim]
    device const T* residual_out_proj,  // [R, input_dim, codebook_dim]
    device const T* residual_out_bias,  // [R, input_dim]
    device T* output,                   // [B, T, input_dim]
    const constant int& batch_size,
    const constant int& total_codebooks,
    const constant int& seq_len,
    const constant int& input_dim,
    const constant int& codebook_dim,
    const constant int& residual_quantizers,
    const constant int& semantic_cardinality,
    const constant int& residual_cardinality,
    const uint3 gid
) {
  const uint d = gid.x;
  const uint t = gid.y;
  const uint b = gid.z;

  if (d >= (uint)input_dim || t >= (uint)seq_len || b >= (uint)batch_size) {
    return;
  }

  const int len_b = lengths ? lengths[b] : seq_len;
  const uint out_idx = (b * (uint)seq_len + t) * (uint)input_dim + d;
  if ((int)t >= len_b) {
    output[out_idx] = (T)0;
    return;
  }

  const uint token_base = (b * (uint)total_codebooks) * (uint)seq_len + t;
  uint semantic_token = tokens[token_base];
  semantic_token = min(semantic_token, (uint)(semantic_cardinality - 1));

  float acc = float(semantic_out_bias[d]);
  const uint semantic_code_base = (uint)semantic_token * (uint)codebook_dim;
  const uint semantic_proj_base = d * (uint)codebook_dim;
  for (uint k = 0; k < (uint)codebook_dim; ++k) {
    acc += float(semantic_out_proj[semantic_proj_base + k]) *
           float(semantic_codebook[semantic_code_base + k]);
  }

  for (int r = 0; r < residual_quantizers; ++r) {
    const uint token_idx = token_base + (uint)(r + 1) * (uint)seq_len;
    uint residual_token = tokens[token_idx];
    residual_token = min(residual_token, (uint)(residual_cardinality - 1));

    const uint bias_idx = (uint)r * (uint)input_dim + d;
    acc += float(residual_out_bias[bias_idx]);

    const uint code_base =
        ((uint)r * (uint)residual_cardinality + (uint)residual_token) *
        (uint)codebook_dim;
    const uint proj_base =
        ((uint)r * (uint)input_dim + d) * (uint)codebook_dim;
    for (uint k = 0; k < (uint)codebook_dim; ++k) {
      acc += float(residual_out_proj[proj_base + k]) *
             float(residual_codebooks[code_base + k]);
    }
  }

  output[out_idx] = (T)acc;
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(AudioQuantizerDecode)(
    device const uint* tokens,
    device const int* lengths,
    device const T* semantic_codebook,
    device const T* semantic_out_proj,
    device const T* semantic_out_bias,
    device const T* residual_codebooks,
    device const T* residual_out_proj,
    device const T* residual_out_bias,
    device T* output,
    const constant int& batch_size,
    const constant int& total_codebooks,
    const constant int& seq_len,
    const constant int& input_dim,
    const constant int& codebook_dim,
    const constant int& residual_quantizers,
    const constant int& semantic_cardinality,
    const constant int& residual_cardinality,
    uint d AXIS(input_dim, 32),
    uint t AXIS(seq_len, 1),
    uint b AXIS(batch_size, 1)
) {
  quantizer_decode<T>(
      tokens,
      lengths,
      semantic_codebook,
      semantic_out_proj,
      semantic_out_bias,
      residual_codebooks,
      residual_out_proj,
      residual_out_bias,
      output,
      batch_size,
      total_codebooks,
      seq_len,
      input_dim,
      codebook_dim,
      residual_quantizers,
      semantic_cardinality,
      residual_cardinality,
      uint3(d, t, b)
  );
}
