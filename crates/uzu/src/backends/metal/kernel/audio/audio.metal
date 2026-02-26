#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

// === FSQ (Finite Scalar Quantization) decode ===
//
// Implements NeMo's FiniteScalarQuantizer.decode() applied per group:
//  - tokens are stored as [B, G, T] int32
//  - output codes are stored as [B, G*D, T] where D = codebook_dim_per_group
//
// NeMo reference:
//   codes_nonnegative = (indices // dim_base_index) % num_levels
//   dequantized = (codes_nonnegative - (num_levels//2)) / (num_levels//2)
//
// Here dim_base_index is derived from num_levels:
//   base[0]=1, base[d]=prod_{k<d} num_levels[k]

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

// === Quantizer decode (semantic + residual VQ) ===
//
// Decodes codebook tokens into latent vectors in NSC layout:
// - tokens: [B, K, T] int32 (codebook-major)
// - output: [B, T, C] float
//
// For each (b, t), this computes:
//   semantic_out_proj @ semantic_codebook[token_0] + semantic_out_bias
// + sum_r residual_out_proj[r] @ residual_codebook[r][token_{r+1}] +
//         residual_out_bias[r]
template <typename T>
void audio_codec_quantizer_decode(
    device const int* tokens,           // [B, K, T]
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
  int semantic_token = tokens[token_base];
  semantic_token = clamp(semantic_token, 0, semantic_cardinality - 1);

  float acc = float(semantic_out_bias[d]);
  const uint semantic_code_base = (uint)semantic_token * (uint)codebook_dim;
  const uint semantic_proj_base = d * (uint)codebook_dim;
  for (uint k = 0; k < (uint)codebook_dim; ++k) {
    acc += float(semantic_out_proj[semantic_proj_base + k]) *
           float(semantic_codebook[semantic_code_base + k]);
  }

  for (int r = 0; r < residual_quantizers; ++r) {
    const uint token_idx = token_base + (uint)(r + 1) * (uint)seq_len;
    int residual_token = tokens[token_idx];
    residual_token = clamp(residual_token, 0, residual_cardinality - 1);

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

// === Layout transpose NSC -> NCS ===
template <typename T>
void audio_codec_transpose_nsc_to_ncs(
    device const T* input, // [B, S, C]
    device T* output,      // [B, C, S]
    const constant int& seq_len,
    const constant int& channels,
    const uint3 gid
) {
  const uint t = gid.x;
  const uint c = gid.y;
  const uint b = gid.z;
  if (t >= (uint)seq_len || c >= (uint)channels) {
    return;
  }

  const uint src_idx = (b * (uint)seq_len + t) * (uint)channels + c;
  const uint dst_idx = (b * (uint)channels + c) * (uint)seq_len + t;
  output[dst_idx] = input[src_idx];
}

// === Elementwise ops for codec blocks ===

template <typename T>
void audio_codec_leaky_relu(
    device const T* input,
    device T* output,
    const constant int& n,
    const constant float& negative_slope,
    const uint tid
) {
  if ((int)tid >= n) {
    return;
  }
  const float x = float(input[tid]);
  const float y = (x >= 0.0f) ? x : (negative_slope * x);
  output[tid] = (T)y;
}

template <typename T>
void audio_codec_tanh(
    device const T* input,
    device T* output,
    const constant int& n,
    const uint tid
) {
  if ((int)tid >= n) {
    return;
  }
  const float x = float(input[tid]);
  output[tid] = (T)tanh(x);
}

template <typename T>
void audio_codec_add(
    device const T* a,
    device const T* b,
    device T* out,
    const constant int& n,
    const uint tid
) {
  if ((int)tid >= n) {
    return;
  }
  out[tid] = a[tid] + b[tid];
}

template <typename T>
void audio_codec_scale(
    device const T* input,
    device T* output,
    const constant int& n,
    const constant float& scale,
    const uint tid
) {
  if ((int)tid >= n) {
    return;
  }
  const float x = float(input[tid]);
  output[tid] = (T)(x * scale);
}

// === Channel-wise norm over NCS tensors ===
//
// Normalizes each [channels] slice at fixed (batch, time) in NCS layout.
// This supports both LayerNorm-style (subtract_mean=1) and RMSNorm-style
// (subtract_mean=0), with optional learned bias passed as a dense vector.
#define AUDIO_NORM_NCS_BLOCK_SIZE 256
#define AUDIO_NORM_NCS_MAX_SIMDS 32

template <typename T>
void audio_codec_norm_ncs(
    device const T* input,     // [B, C, T]
    device const T* scales,    // [C]
    device const T* bias,      // [C]
    device T* output,          // [B, C, T]
    device const int* lengths, // [B]
    const constant int& channels,
    const constant int& seq_len,
    const constant float& epsilon,
    const constant int& subtract_mean,
    threadgroup float* shared_mean,
    threadgroup float* shared_variance,
    const thread Simd& simd,
    const uint b,
    const uint t,
    const uint lid
) {
  if (t >= (uint)seq_len || channels <= 0) {
    return;
  }

  const int len_b = lengths ? lengths[b] : seq_len;

  if ((int)t >= len_b) {
    for (uint c = lid; c < (uint)channels; c += AUDIO_NORM_NCS_BLOCK_SIZE) {
      const uint out_idx = (b * (uint)channels + c) * (uint)seq_len + t;
      output[out_idx] = (T)0;
    }
    return;
  }

  float partial_sum = 0.0f;
  if (subtract_mean != 0) {
    for (uint c = lid; c < (uint)channels; c += AUDIO_NORM_NCS_BLOCK_SIZE) {
      const uint idx = (b * (uint)channels + c) * (uint)seq_len + t;
      partial_sum += float(input[idx]);
    }
  }

  const float sum = threadgroup_cooperative_reduce_sum<AUDIO_NORM_NCS_BLOCK_SIZE>(
      partial_sum,
      shared_mean,
      (ushort)lid,
      simd
  );
  const float mean = (subtract_mean != 0) ? (sum / (float)channels) : 0.0f;

  float partial_variance = 0.0f;
  for (uint c = lid; c < (uint)channels; c += AUDIO_NORM_NCS_BLOCK_SIZE) {
    const uint idx = (b * (uint)channels + c) * (uint)seq_len + t;
    const float x = float(input[idx]);
    const float centered = (subtract_mean != 0) ? (x - mean) : x;
    partial_variance += centered * centered;
  }

  const float variance_sum = threadgroup_cooperative_reduce_sum<AUDIO_NORM_NCS_BLOCK_SIZE>(
      partial_variance,
      shared_variance,
      (ushort)lid,
      simd
  );
  const float inv_std = rsqrt(variance_sum / (float)channels + epsilon);

  for (uint c = lid; c < (uint)channels; c += AUDIO_NORM_NCS_BLOCK_SIZE) {
    const uint idx = (b * (uint)channels + c) * (uint)seq_len + t;
    const float x = float(input[idx]);
    const float centered = (subtract_mean != 0) ? (x - mean) : x;
    const float y = centered * inv_std * float(scales[c]) + float(bias[c]);
    output[idx] = (T)y;
  }
}

// === Causal Conv1d (stride=1, groups=1) ===
//
// Matches NeMo's CausalConv1dNorm forward for the decoder path where
// stride=1:
// - left padding by padding_total = (kernel_size_eff - 1) where
// kernel_size_eff=(K-1)*dilation+1
// - no right padding (extra_padding=0 for stride=1)
// - output length equals input length
template <typename T>
void audio_codec_causal_conv1d(
    device const T* input,     // [B, Cin, T]
    device const T* weight,    // [Cout, Cin, K]
    device const T* bias,      // [Cout]
    device T* output,          // [B, Cout, T]
    device const int* lengths, // [B]
    const constant int& cin,
    const constant int& cout,
    const constant int& seq_len,
    const constant int& kernel_size,
    const constant int& dilation,
    const uint3 gid
) {
  const uint t = gid.x;
  const uint oc = gid.y;
  const uint b = gid.z;

  if (t >= (uint)seq_len || oc >= (uint)cout) {
    return;
  }

  const int len_b = lengths ? lengths[b] : seq_len;
  const uint out_idx = (b * (uint)cout + oc) * (uint)seq_len + t;
  if ((int)t >= len_b) {
    output[out_idx] = (T)0;
    return;
  }

  const int pad = (kernel_size - 1) * dilation;
  float acc = float(bias[oc]);

  for (int ic = 0; ic < cin; ++ic) {
    const uint w_base = (oc * (uint)cin + (uint)ic) * (uint)kernel_size;
    const uint x_base = (b * (uint)cin + (uint)ic) * (uint)seq_len;
    for (int k = 0; k < kernel_size; ++k) {
      const int x_t = (int)t + k * dilation - pad;
      if (x_t < 0) {
        continue;
      }
      // For stride=1 and causal padding, x_t never exceeds len_b for valid t,
      // but for safety clamp to seq_len.
      if (x_t >= seq_len) {
        continue;
      }
      acc += float(weight[w_base + (uint)k]) * float(input[x_base + (uint)x_t]);
    }
  }

  output[out_idx] = (T)acc;
}

// === Grouped Causal Conv1d (stride=1, groups>=1) ===
//
// Same semantics as audio_codec_causal_conv1d, but with grouped channel
// partitions. Weight layout is [Cout, Cin/groups, K].
template <typename T>
void audio_codec_causal_conv1d_grouped(
    device const T* input,     // [B, Cin, T]
    device const T* weight,    // [Cout, Cin/groups, K]
    device const T* bias,      // [Cout]
    device T* output,          // [B, Cout, T]
    device const int* lengths, // [B]
    const constant int& cin,
    const constant int& cout,
    const constant int& seq_len,
    const constant int& kernel_size,
    const constant int& dilation,
    const constant int& groups,
    const uint3 gid
) {
  const uint t = gid.x;
  const uint oc = gid.y;
  const uint b = gid.z;

  if (t >= (uint)seq_len || oc >= (uint)cout) {
    return;
  }

  if (groups <= 0 || (cin % groups) != 0 || (cout % groups) != 0) {
    return;
  }

  const int len_b = lengths ? lengths[b] : seq_len;
  const uint out_idx = (b * (uint)cout + oc) * (uint)seq_len + t;
  if ((int)t >= len_b) {
    output[out_idx] = (T)0;
    return;
  }

  const int cin_per_group = cin / groups;
  const int cout_per_group = cout / groups;
  const int group_idx = (int)oc / cout_per_group;
  const int oc_in_group = (int)oc - group_idx * cout_per_group;
  const int in_begin = group_idx * cin_per_group;

  const int pad = (kernel_size - 1) * dilation;
  float acc = float(bias[oc]);

  for (int ic_local = 0; ic_local < cin_per_group; ++ic_local) {
    const int ic = in_begin + ic_local;
    const uint w_base =
        ((group_idx * cout_per_group + oc_in_group) * (uint)cin_per_group +
         (uint)ic_local) *
        (uint)kernel_size;
    const uint x_base = (b * (uint)cin + (uint)ic) * (uint)seq_len;

    for (int k = 0; k < kernel_size; ++k) {
      const int x_t = (int)t + k * dilation - pad;
      if (x_t < 0 || x_t >= seq_len) {
        continue;
      }
      acc += float(weight[w_base + (uint)k]) * float(input[x_base + (uint)x_t]);
    }
  }

  output[out_idx] = (T)acc;
}

// === Causal ConvTranspose1d upsample (kernel_size = 2*stride) ===
//
// Matches NeMo's CausalConvTranspose1dNorm forward for trim_right_ratio=1:
// - ConvTranspose1d with padding=0, output_padding=0, dilation=1
// - kernel_size = 2 * stride
// - unpad: trim_right = kernel_size - stride (= stride), trim_left = 0
// Resulting output length is input_len * stride.
template <typename T>
void audio_codec_causal_conv_transpose1d(
    device const T* input,     // [B, Cin, Tin]
    device const T* weight,    // [Cin, Cout_per_group, 2*stride]
    device const T* bias,      // [Cout]
    device T* output,          // [B, Cout, Tout]
    device const int* lengths, // [B] (output lengths)
    const constant int& cin,
    const constant int& cout,
    const constant int& seq_len_in,
    const constant int& seq_len_out,
    const constant int& stride,
    const constant int& groups,
    const uint3 gid
) {
  const uint t_out = gid.x;
  const uint oc = gid.y;
  const uint b = gid.z;

  if (t_out >= (uint)seq_len_out || oc >= (uint)cout) {
    return;
  }

  const int len_b = lengths ? lengths[b] : seq_len_out;
  const uint out_idx = (b * (uint)cout + oc) * (uint)seq_len_out + t_out;
  if ((int)t_out >= len_b) {
    output[out_idx] = (T)0;
    return;
  }

  const int cout_per_group = cout / groups;
  const int cin_per_group = cin / groups;
  const int group_idx = (int)oc / cout_per_group;
  const int oc_in_group = (int)oc - group_idx * cout_per_group;

  const int q = (int)t_out / stride;
  const int r = (int)t_out - q * stride;

  float acc = float(bias[oc]);

  const uint in_base_b = b * (uint)cin * (uint)seq_len_in;
  const uint w_base_oc = (uint)oc_in_group * (uint)(2 * stride);

  const int ic_begin = group_idx * cin_per_group;
  const int ic_end = ic_begin + cin_per_group;

  for (int ic = ic_begin; ic < ic_end; ++ic) {
    if (q >= seq_len_in) {
      continue;
    }
    const uint in_base = in_base_b + (uint)ic * (uint)seq_len_in;

    // Weight layout: [Cin, Cout_per_group, K]
    const uint w_base =
        ((uint)ic * (uint)cout_per_group) * (uint)(2 * stride) + w_base_oc;

    // Contribution from input[q] with k=r
    acc += float(weight[w_base + (uint)r]) * float(input[in_base + (uint)q]);

    // Contribution from input[q-1] with k=stride+r
    if (q > 0) {
      acc += float(weight[w_base + (uint)(stride + r)]) *
             float(input[in_base + (uint)(q - 1)]);
    }
  }

  output[out_idx] = (T)acc;
}

// Input layout for audio_codec_causal_conv_transpose1d_lalamo:
// 0 = NCS ([B, Cin, Tin]), 1 = NSC ([B, Tin, Cin]).
constant int AUDIO_LAYOUT_NCS = 0;
constant int AUDIO_LAYOUT_NSC = 1;

// === Causal ConvTranspose1d (Lalamo semantics, arbitrary kernel_size) ===
//
// This matches causal_conv_transpose1d_lalamo_reference():
// - Expand input by stride with zeros inserted between samples.
// - Left pad by (kernel_size - 1), no right pad.
// - Convolve with provided kernel along expanded timeline.
template <typename T>
void audio_codec_causal_conv_transpose1d_lalamo(
    device const T* input,     // [B, Cin, Tin] when NCS, [B, Tin, Cin] when NSC
    device const T* weight,    // [Cin, Cout_per_group, K]
    device const T* bias,      // [Cout]
    device T* output,          // [B, Cout, Tout]
    device const int* lengths, // [B]
    const constant int& cin,
    const constant int& cout,
    const constant int& seq_len_in,
    const constant int& seq_len_out,
    const constant int& kernel_size,
    const constant int& stride,
    const constant int& groups,
    const constant int& input_layout,
    const uint3 gid
) {
  const uint t_out = gid.x;
  const uint oc = gid.y;
  const uint b = gid.z;

  if (t_out >= (uint)seq_len_out || oc >= (uint)cout) {
    return;
  }
  if ((input_layout != AUDIO_LAYOUT_NCS && input_layout != AUDIO_LAYOUT_NSC) ||
      groups <= 0 || stride <= 0 || kernel_size <= 0 || (cin % groups) != 0 ||
      (cout % groups) != 0) {
    return;
  }

  const int len_b = lengths ? lengths[b] : seq_len_out;
  const uint out_idx = (b * (uint)cout + oc) * (uint)seq_len_out + t_out;
  if ((int)t_out >= len_b) {
    output[out_idx] = (T)0;
    return;
  }

  const int cout_per_group = cout / groups;
  const int cin_per_group = cin / groups;
  const int group_idx = (int)oc / cout_per_group;
  const int oc_in_group = (int)oc - group_idx * cout_per_group;
  const int ic_begin = group_idx * cin_per_group;
  const int ic_end = ic_begin + cin_per_group;

  const int seq_len_expanded =
      (seq_len_in > 0) ? (((seq_len_in - 1) * stride) + 1) : 0;
  const int left_pad = kernel_size - 1;

  float acc = float(bias[oc]);
  if (input_layout == AUDIO_LAYOUT_NCS) {
    for (int ic = ic_begin; ic < ic_end; ++ic) {
      const uint in_base = (b * (uint)cin + (uint)ic) * (uint)seq_len_in;
      const uint w_base =
          ((uint)ic * (uint)cout_per_group + (uint)oc_in_group) *
          (uint)kernel_size;

      for (int k = 0; k < kernel_size; ++k) {
        const int expanded_time = (int)t_out + k - left_pad;
        if (expanded_time < 0 || expanded_time >= seq_len_expanded) {
          continue;
        }
        if ((expanded_time % stride) != 0) {
          continue;
        }

        const int src_time = expanded_time / stride;
        if (src_time < 0 || src_time >= seq_len_in) {
          continue;
        }
        acc += float(input[in_base + (uint)src_time]) *
               float(weight[w_base + (uint)k]);
      }
    }
  } else {
    for (int ic = ic_begin; ic < ic_end; ++ic) {
      const uint w_base =
          ((uint)ic * (uint)cout_per_group + (uint)oc_in_group) *
          (uint)kernel_size;

      for (int k = 0; k < kernel_size; ++k) {
        const int expanded_time = (int)t_out + k - left_pad;
        if (expanded_time < 0 || expanded_time >= seq_len_expanded) {
          continue;
        }
        if ((expanded_time % stride) != 0) {
          continue;
        }

        const int src_time = expanded_time / stride;
        if (src_time < 0 || src_time >= seq_len_in) {
          continue;
        }
        const uint in_idx =
            (b * (uint)seq_len_in + (uint)src_time) * (uint)cin + (uint)ic;
        acc += float(input[in_idx]) * float(weight[w_base + (uint)k]);
      }
    }
  }

  output[out_idx] = (T)acc;
}

// === Snake / HalfSnake / Clamp (NeMo common.parts.utils) ===
//
// snake(x, alpha, eps) = x + (alpha + eps)^-1 * sin(alpha * x)^2
// HalfSnake applies snake to the first half channels and LeakyReLU to the
// rest.

template <typename T>
void audio_codec_half_snake(
    device const T* input, // [B, C, T]
    device const T* alpha, // [1, C_snake, 1] (contiguous, index by channel)
    device T* output,      // [B, C, T]
    const constant int& channels,
    const constant int& seq_len,
    const constant int& snake_channels,
    const constant float& negative_slope,
    const constant float& eps,
    const uint3 gid
) {
  const uint t = gid.x;
  const uint c = gid.y;
  const uint b = gid.z;

  if (t >= (uint)seq_len || c >= (uint)channels) {
    return;
  }

  const uint idx = (b * (uint)channels + c) * (uint)seq_len + t;
  const float x = float(input[idx]);

  if ((int)c < snake_channels) {
    const float a = float(alpha[c]);
    const float ax = a * x;
    const float s = sin(ax);
    const float y = x + (1.0f / (a + eps)) * (s * s);
    output[idx] = (T)y;
  } else {
    const float y = (x >= 0.0f) ? x : (negative_slope * x);
    output[idx] = (T)y;
  }
}

template <typename T>
void audio_codec_clamp(
    device const T* input,
    device T* output,
    const constant int& n,
    const constant float& min_value,
    const constant float& max_value,
    const uint tid
) {
  if ((int)tid >= n) {
    return;
  }
  const float x = float(input[tid]);
  const float y = clamp(x, min_value, max_value);
  output[tid] = (T)y;
}

// === Non-causal Conv1d with padding + stride (for encoder) ===
//
// Matches torch.nn.Conv1d with explicit padding and padding_mode:
// out[t_out] = bias + sum_{ic,k} w[oc,ic,k] * x[ic, t_out*stride - padding
// + k*dilation]
// - pad_mode=0: zeros
// - pad_mode=1: replicate (clamp index to [0, seq_len_in-1])
template <typename T>
void audio_codec_conv1d(
    device const T* input,     // [B, Cin, Tin]
    device const T* weight,    // [Cout, Cin, K]
    device const T* bias,      // [Cout]
    device T* output,          // [B, Cout, Tout]
    device const int* lengths, // [B] (Tout lengths)
    const constant int& cin,
    const constant int& cout,
    const constant int& seq_len_in,
    const constant int& seq_len_out,
    const constant int& kernel_size,
    const constant int& stride,
    const constant int& dilation,
    const constant int& padding,
    const constant int& pad_mode,
    const uint3 gid
) {
  const uint t_out = gid.x;
  const uint oc = gid.y;
  const uint b = gid.z;

  if (t_out >= (uint)seq_len_out || oc >= (uint)cout) {
    return;
  }

  const int len_b = lengths ? lengths[b] : seq_len_out;
  const uint out_idx = (b * (uint)cout + oc) * (uint)seq_len_out + t_out;
  if ((int)t_out >= len_b) {
    output[out_idx] = (T)0;
    return;
  }

  const int base = (int)t_out * stride - padding;
  float acc = float(bias[oc]);

  for (int ic = 0; ic < cin; ++ic) {
    const uint w_base = (oc * (uint)cin + (uint)ic) * (uint)kernel_size;
    const uint x_base = (b * (uint)cin + (uint)ic) * (uint)seq_len_in;
    for (int k = 0; k < kernel_size; ++k) {
      int x_t = base + k * dilation;
      if (pad_mode == 0) {
        if (x_t < 0 || x_t >= seq_len_in) {
          continue;
        }
      } else {
        x_t = clamp(x_t, 0, seq_len_in - 1);
      }
      acc += float(weight[w_base + (uint)k]) * float(input[x_base + (uint)x_t]);
    }
  }

  output[out_idx] = (T)acc;
}

// === FSQ (Finite Scalar Quantization) encode ===
//
// Implements GroupFiniteScalarQuantizer.encode() for inference:
// - input:  [B, G*D, T] float
// - output: [B, G,   T] int32
//
// NeMo reference (FiniteScalarQuantizer):
//   output_scale = (num_levels - 1) / 2 * (1 - eps)
//   output_offset = 0.5 if even else 0
//   input_shift = tan(output_offset / output_scale)
//   compressed = output_scale * tanh(inputs + input_shift) - output_offset
//   rounded = round(compressed)  # half away from zero
//   code_nonneg = rounded + (num_levels//2)
//   token = sum_d code_nonneg[d] * dim_base_index[d]

// Match torch.round() semantics (ties to even).
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

// === DSL kernel wrappers ===
//
// Keep the scalar helper implementations above and expose a DSL-annotated
// surface so audio kernels are integrated through the same generated wrapper
// path as other kernels.

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

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioQuantizerDecode)(
    device const int* tokens,
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
  audio_codec_quantizer_decode<T>(
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

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioFishAudioQuantizerDecode)(
    device const int* tokens,
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
  audio_codec_quantizer_decode<T>(
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

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioTransposeNscToNcs)(
    device const T* input,
    device T* output,
    const constant int& seq_len,
    const constant int& channels,
    const constant int& batch_size,
    uint t AXIS(seq_len, 32),
    uint c AXIS(channels, 1),
    uint b AXIS(batch_size, 1)
) {
  audio_codec_transpose_nsc_to_ncs<T>(
      input,
      output,
      seq_len,
      channels,
      uint3(t, c, b)
  );
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioLeakyRelu)(
    device const T* input,
    device T* output,
    const constant int& n,
    const constant float& negative_slope,
    uint tid AXIS(n, 256)
) {
  audio_codec_leaky_relu<T>(input, output, n, negative_slope, tid);
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioTanh)(
    device const T* input,
    device T* output,
    const constant int& n,
    uint tid AXIS(n, 256)
) {
  audio_codec_tanh<T>(input, output, n, tid);
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioAdd)(
    device const T* a,
    device const T* b,
    device T* out,
    const constant int& n,
    uint tid AXIS(n, 256)
) {
  audio_codec_add<T>(a, b, out, n, tid);
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioScale)(
    device const T* input,
    device T* output,
    const constant int& n,
    const constant float& scale,
    uint tid AXIS(n, 256)
) {
  audio_codec_scale<T>(input, output, n, scale, tid);
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioNormNcs)(
    device const T* input,
    device const T* scales,
    device const T* bias,
    device T* output,
    device const int* lengths,
    const constant int& channels,
    const constant int& seq_len,
    const constant float& epsilon,
    const constant int& subtract_mean,
    const constant int& batch_size,
    threadgroup float shared_mean[AUDIO_NORM_NCS_MAX_SIMDS],
    threadgroup float shared_variance[AUDIO_NORM_NCS_MAX_SIMDS],
    const Simd simd,
    const uint b GROUPS(batch_size),
    const uint t GROUPS(seq_len),
    const uint lid THREADS(AUDIO_NORM_NCS_BLOCK_SIZE)
) {
  audio_codec_norm_ncs<T>(
      input,
      scales,
      bias,
      output,
      lengths,
      channels,
      seq_len,
      epsilon,
      subtract_mean,
      shared_mean,
      shared_variance,
      simd,
      b,
      t,
      lid
  );
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioCausalConv1d)(
    device const T* input,
    device const T* weight,
    device const T* bias,
    device T* output,
    device const int* lengths,
    const constant int& cin,
    const constant int& cout,
    const constant int& seq_len,
    const constant int& kernel_size,
    const constant int& dilation,
    const constant int& batch_size,
    uint t AXIS(seq_len, 32),
    uint oc AXIS(cout, 1),
    uint b AXIS(batch_size, 1)
) {
  audio_codec_causal_conv1d<T>(
      input,
      weight,
      bias,
      output,
      lengths,
      cin,
      cout,
      seq_len,
      kernel_size,
      dilation,
      uint3(t, oc, b)
  );
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioCausalConv1dGrouped)(
    device const T* input,
    device const T* weight,
    device const T* bias,
    device T* output,
    device const int* lengths,
    const constant int& cin,
    const constant int& cout,
    const constant int& seq_len,
    const constant int& kernel_size,
    const constant int& dilation,
    const constant int& groups,
    const constant int& batch_size,
    uint t AXIS(seq_len, 32),
    uint oc AXIS(cout, 1),
    uint b AXIS(batch_size, 1)
) {
  audio_codec_causal_conv1d_grouped<T>(
      input,
      weight,
      bias,
      output,
      lengths,
      cin,
      cout,
      seq_len,
      kernel_size,
      dilation,
      groups,
      uint3(t, oc, b)
  );
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioCausalConvTranspose1d)(
    device const T* input,
    device const T* weight,
    device const T* bias,
    device T* output,
    device const int* lengths,
    const constant int& cin,
    const constant int& cout,
    const constant int& seq_len_in,
    const constant int& seq_len_out,
    const constant int& stride,
    const constant int& groups,
    const constant int& batch_size,
    uint t_out AXIS(seq_len_out, 32),
    uint oc AXIS(cout, 1),
    uint b AXIS(batch_size, 1)
) {
  audio_codec_causal_conv_transpose1d<T>(
      input,
      weight,
      bias,
      output,
      lengths,
      cin,
      cout,
      seq_len_in,
      seq_len_out,
      stride,
      groups,
      uint3(t_out, oc, b)
  );
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioCausalConvTranspose1dLalamo)(
    device const T* input,
    device const T* weight,
    device const T* bias,
    device T* output,
    device const int* lengths,
    const constant int& cin,
    const constant int& cout,
    const constant int& seq_len_in,
    const constant int& seq_len_out,
    const constant int& kernel_size,
    const constant int& stride,
    const constant int& groups,
    const constant int& input_layout,
    const constant int& batch_size,
    uint t_out AXIS(seq_len_out, 32),
    uint oc AXIS(cout, 1),
    uint b AXIS(batch_size, 1)
) {
  audio_codec_causal_conv_transpose1d_lalamo<T>(
      input,
      weight,
      bias,
      output,
      lengths,
      cin,
      cout,
      seq_len_in,
      seq_len_out,
      kernel_size,
      stride,
      groups,
      input_layout,
      uint3(t_out, oc, b)
  );
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioHalfSnake)(
    device const T* input,
    device const T* alpha,
    device T* output,
    const constant int& channels,
    const constant int& seq_len,
    const constant int& snake_channels,
    const constant float& negative_slope,
    const constant float& eps,
    const constant int& batch_size,
    uint t AXIS(seq_len, 32),
    uint c AXIS(channels, 1),
    uint b AXIS(batch_size, 1)
) {
  audio_codec_half_snake<T>(
      input,
      alpha,
      output,
      channels,
      seq_len,
      snake_channels,
      negative_slope,
      eps,
      uint3(t, c, b)
  );
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioClamp)(
    device const T* input,
    device T* output,
    const constant int& n,
    const constant float& min_value,
    const constant float& max_value,
    uint tid AXIS(n, 256)
) {
  audio_codec_clamp<T>(input, output, n, min_value, max_value, tid);
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioConv1d)(
    device const T* input,
    device const T* weight,
    device const T* bias,
    device T* output,
    device const int* lengths,
    const constant int& cin,
    const constant int& cout,
    const constant int& seq_len_in,
    const constant int& seq_len_out,
    const constant int& kernel_size,
    const constant int& stride,
    const constant int& dilation,
    const constant int& padding,
    const constant int& pad_mode,
    const constant int& batch_size,
    uint t_out AXIS(seq_len_out, 32),
    uint oc AXIS(cout, 1),
    uint b AXIS(batch_size, 1)
) {
  audio_codec_conv1d<T>(
      input,
      weight,
      bias,
      output,
      lengths,
      cin,
      cout,
      seq_len_in,
      seq_len_out,
      kernel_size,
      stride,
      dilation,
      padding,
      pad_mode,
      uint3(t_out, oc, b)
  );
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
