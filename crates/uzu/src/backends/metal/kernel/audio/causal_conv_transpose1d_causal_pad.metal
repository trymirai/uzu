#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

// Input layout for causal_conv_transpose1d_causal_pad:
// 0 = NCS ([B, Cin, Tin]), 1 = NSC ([B, Tin, Cin]).
constant int AUDIO_LAYOUT_NCS = 0;
constant int AUDIO_LAYOUT_NSC = 1;

// === Causal ConvTranspose1d (causal-pad semantics, arbitrary kernel_size) ===
//
// This matches causal_conv_transpose1d_causal_pad_reference():
// - Expand input by stride with zeros inserted between samples.
// - Left pad by (kernel_size - 1), no right pad.
// - Convolve with provided kernel along expanded timeline.
template <typename T>
void causal_conv_transpose1d_causal_pad(
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
  const bool fast_two_tap = (kernel_size == (stride * 2));
  const int q = (int)t_out / stride;
  const int r = (int)t_out % stride;
  const int k_q_minus_one = (stride - 1) - r;
  const int k_q = (kernel_size - 1) - r;

  float acc = float(bias[oc]);
  if (input_layout == AUDIO_LAYOUT_NCS) {
    for (int ic = ic_begin; ic < ic_end; ++ic) {
      const uint in_base = (b * (uint)cin + (uint)ic) * (uint)seq_len_in;
      const uint w_base =
          ((uint)ic * (uint)cout_per_group + (uint)oc_in_group) *
          (uint)kernel_size;
      if (fast_two_tap) {
        if (q > 0 && q - 1 < seq_len_in) {
          acc += float(input[in_base + (uint)(q - 1)]) *
                 float(weight[w_base + (uint)k_q_minus_one]);
        }
        if (q >= 0 && q < seq_len_in) {
          acc += float(input[in_base + (uint)q]) *
                 float(weight[w_base + (uint)k_q]);
        }
      } else {
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
    }
  } else {
    for (int ic = ic_begin; ic < ic_end; ++ic) {
      const uint w_base =
          ((uint)ic * (uint)cout_per_group + (uint)oc_in_group) *
          (uint)kernel_size;
      if (fast_two_tap) {
        if (q > 0 && q - 1 < seq_len_in) {
          const uint in_idx = (b * (uint)seq_len_in + (uint)(q - 1)) * (uint)cin + (uint)ic;
          acc += float(input[in_idx]) * float(weight[w_base + (uint)k_q_minus_one]);
        }
        if (q >= 0 && q < seq_len_in) {
          const uint in_idx = (b * (uint)seq_len_in + (uint)q) * (uint)cin + (uint)ic;
          acc += float(input[in_idx]) * float(weight[w_base + (uint)k_q]);
        }
      } else {
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
  }

  output[out_idx] = (T)acc;
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(AudioCausalConvTranspose1dCausalPad)(
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
  causal_conv_transpose1d_causal_pad<T>(
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
