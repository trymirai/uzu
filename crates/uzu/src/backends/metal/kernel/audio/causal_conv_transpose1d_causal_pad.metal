#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

constant uint AUDIO_TIME_TILE = 4;

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
    device const T* weight,    // [Cout, Cin_per_group, K]
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
  const uint t_out = gid.x * AUDIO_TIME_TILE;
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
  const uint out_base = (b * (uint)cout + oc) * (uint)seq_len_out + t_out;
  int lane_count = seq_len_out - (int)t_out;
  if (lane_count > (int)AUDIO_TIME_TILE) {
    lane_count = (int)AUDIO_TIME_TILE;
  }
  if (lane_count <= 0) {
    return;
  }
  int valid_count = len_b - (int)t_out;
  if (valid_count > lane_count) {
    valid_count = lane_count;
  }
  if (valid_count < 0) {
    valid_count = 0;
  }

  const int cout_per_group = cout / groups;
  const int cin_per_group = cin / groups;
  const int group_idx = (int)oc / cout_per_group;
  const int ic_begin = group_idx * cin_per_group;
  const int ic_end = ic_begin + cin_per_group;

  const int seq_len_expanded =
      (seq_len_in > 0) ? (((seq_len_in - 1) * stride) + 1) : 0;
  const int left_pad = kernel_size - 1;
  const bool fast_two_tap = (kernel_size == (stride * 2));

  const float bias_value = float(bias[oc]);
  float acc0 = bias_value;
  float acc1 = bias_value;
  float acc2 = bias_value;
  float acc3 = bias_value;

  // Process each lane that is valid (within sequence length)
  if (valid_count > 0) {
    // Precompute q and r for lane 0; consecutive lanes have t_out+1, t_out+2,
    // t_out+3.
    const int t0 = (int)t_out;
    const int q0 = t0 / stride;
    const int r0 = t0 % stride;

    if (input_layout == AUDIO_LAYOUT_NCS) {
      for (int ic = ic_begin; ic < ic_end; ++ic) {
        const uint in_base = (b * (uint)cin + (uint)ic) * (uint)seq_len_in;
        const uint ic_local = (uint)(ic - ic_begin);
        const uint w_base =
            ((uint)oc * (uint)cin_per_group + ic_local) * (uint)kernel_size;
        if (fast_two_tap) {
          // When stride >= AUDIO_TIME_TILE (common for decoder blocks with
          // strides 4,8), all 4 lanes share the same q value. This means
          // they all read from the same input positions (q and q-1), just
          // with different weight indices. We load each input value once
          // and broadcast it across lanes.
          if (stride >= (int)AUDIO_TIME_TILE) {
            // All lanes have same q since t_out+3 < t_out+stride means
            // (t_out+3)/stride == t_out/stride.
            const float in_qm1 = (q0 > 0 && q0 - 1 < seq_len_in)
                                     ? float(input[in_base + (uint)(q0 - 1)])
                                     : 0.0f;
            const float in_q0 = (q0 >= 0 && q0 < seq_len_in)
                                    ? float(input[in_base + (uint)q0])
                                    : 0.0f;
            // Lane 0: r = r0
            if (valid_count > 0) {
              acc0 +=
                  in_qm1 * float(weight[w_base + (uint)((stride - 1) - r0)]);
              acc0 += in_q0 *
                      float(weight[w_base + (uint)((kernel_size - 1) - r0)]);
            }
            // Lane 1: r = r0+1
            if (valid_count > 1) {
              const int r1 = r0 + 1;
              acc1 +=
                  in_qm1 * float(weight[w_base + (uint)((stride - 1) - r1)]);
              acc1 += in_q0 *
                      float(weight[w_base + (uint)((kernel_size - 1) - r1)]);
            }
            // Lane 2: r = r0+2
            if (valid_count > 2) {
              const int r2 = r0 + 2;
              acc2 +=
                  in_qm1 * float(weight[w_base + (uint)((stride - 1) - r2)]);
              acc2 += in_q0 *
                      float(weight[w_base + (uint)((kernel_size - 1) - r2)]);
            }
            // Lane 3: r = r0+3
            if (valid_count > 3) {
              const int r3 = r0 + 3;
              acc3 +=
                  in_qm1 * float(weight[w_base + (uint)((stride - 1) - r3)]);
              acc3 += in_q0 *
                      float(weight[w_base + (uint)((kernel_size - 1) - r3)]);
            }
          } else {
            // stride < 4 (e.g., stride=2 for upsampler blocks).
            // Lanes may cross stride boundaries (different q values).
            if (valid_count > 0) {
              const int k_lo = (stride - 1) - r0;
              const int k_hi = (kernel_size - 1) - r0;
              if (q0 > 0 && q0 - 1 < seq_len_in) {
                acc0 += float(input[in_base + (uint)(q0 - 1)]) *
                        float(weight[w_base + (uint)k_lo]);
              }
              if (q0 >= 0 && q0 < seq_len_in) {
                acc0 += float(input[in_base + (uint)q0]) *
                        float(weight[w_base + (uint)k_hi]);
              }
            }
            if (valid_count > 1) {
              const int q1 = (t0 + 1) / stride;
              const int r1 = (t0 + 1) % stride;
              if (q1 > 0 && q1 - 1 < seq_len_in) {
                acc1 += float(input[in_base + (uint)(q1 - 1)]) *
                        float(weight[w_base + (uint)((stride - 1) - r1)]);
              }
              if (q1 >= 0 && q1 < seq_len_in) {
                acc1 += float(input[in_base + (uint)q1]) *
                        float(weight[w_base + (uint)((kernel_size - 1) - r1)]);
              }
            }
            if (valid_count > 2) {
              const int q2 = (t0 + 2) / stride;
              const int r2 = (t0 + 2) % stride;
              if (q2 > 0 && q2 - 1 < seq_len_in) {
                acc2 += float(input[in_base + (uint)(q2 - 1)]) *
                        float(weight[w_base + (uint)((stride - 1) - r2)]);
              }
              if (q2 >= 0 && q2 < seq_len_in) {
                acc2 += float(input[in_base + (uint)q2]) *
                        float(weight[w_base + (uint)((kernel_size - 1) - r2)]);
              }
            }
            if (valid_count > 3) {
              const int q3 = (t0 + 3) / stride;
              const int r3 = (t0 + 3) % stride;
              if (q3 > 0 && q3 - 1 < seq_len_in) {
                acc3 += float(input[in_base + (uint)(q3 - 1)]) *
                        float(weight[w_base + (uint)((stride - 1) - r3)]);
              }
              if (q3 >= 0 && q3 < seq_len_in) {
                acc3 += float(input[in_base + (uint)q3]) *
                        float(weight[w_base + (uint)((kernel_size - 1) - r3)]);
              }
            }
          }
        } else {
          for (int k = 0; k < kernel_size; ++k) {
            const float w = float(weight[w_base + (uint)k]);
            // Lane 0
            if (valid_count > 0) {
              const int expanded_time0 = (int)t_out + k - left_pad;
              if (expanded_time0 >= 0 && expanded_time0 < seq_len_expanded &&
                  (expanded_time0 % stride) == 0) {
                const int src_time0 = expanded_time0 / stride;
                if (src_time0 >= 0 && src_time0 < seq_len_in) {
                  acc0 += float(input[in_base + (uint)src_time0]) * w;
                }
              }
            }
            // Lane 1
            if (valid_count > 1) {
              const int expanded_time1 = (int)t_out + 1 + k - left_pad;
              if (expanded_time1 >= 0 && expanded_time1 < seq_len_expanded &&
                  (expanded_time1 % stride) == 0) {
                const int src_time1 = expanded_time1 / stride;
                if (src_time1 >= 0 && src_time1 < seq_len_in) {
                  acc1 += float(input[in_base + (uint)src_time1]) * w;
                }
              }
            }
            // Lane 2
            if (valid_count > 2) {
              const int expanded_time2 = (int)t_out + 2 + k - left_pad;
              if (expanded_time2 >= 0 && expanded_time2 < seq_len_expanded &&
                  (expanded_time2 % stride) == 0) {
                const int src_time2 = expanded_time2 / stride;
                if (src_time2 >= 0 && src_time2 < seq_len_in) {
                  acc2 += float(input[in_base + (uint)src_time2]) * w;
                }
              }
            }
            // Lane 3
            if (valid_count > 3) {
              const int expanded_time3 = (int)t_out + 3 + k - left_pad;
              if (expanded_time3 >= 0 && expanded_time3 < seq_len_expanded &&
                  (expanded_time3 % stride) == 0) {
                const int src_time3 = expanded_time3 / stride;
                if (src_time3 >= 0 && src_time3 < seq_len_in) {
                  acc3 += float(input[in_base + (uint)src_time3]) * w;
                }
              }
            }
          }
        }
      }
    } else {
      // NSC layout
      for (int ic = ic_begin; ic < ic_end; ++ic) {
        const uint ic_local = (uint)(ic - ic_begin);
        const uint w_base =
            ((uint)oc * (uint)cin_per_group + ic_local) * (uint)kernel_size;
        if (fast_two_tap) {
          if (stride >= (int)AUDIO_TIME_TILE) {
            // All lanes share q0; load input once per (q-1, q) pair.
            const float in_qm1 =
                (q0 > 0 && q0 - 1 < seq_len_in)
                    ? float(input
                                [(b * (uint)seq_len_in + (uint)(q0 - 1)) *
                                     (uint)cin +
                                 (uint)ic])
                    : 0.0f;
            const float in_q0 =
                (q0 >= 0 && q0 < seq_len_in)
                    ? float(input
                                [(b * (uint)seq_len_in + (uint)q0) * (uint)cin +
                                 (uint)ic])
                    : 0.0f;
            if (valid_count > 0) {
              acc0 +=
                  in_qm1 * float(weight[w_base + (uint)((stride - 1) - r0)]);
              acc0 += in_q0 *
                      float(weight[w_base + (uint)((kernel_size - 1) - r0)]);
            }
            if (valid_count > 1) {
              const int r1 = r0 + 1;
              acc1 +=
                  in_qm1 * float(weight[w_base + (uint)((stride - 1) - r1)]);
              acc1 += in_q0 *
                      float(weight[w_base + (uint)((kernel_size - 1) - r1)]);
            }
            if (valid_count > 2) {
              const int r2 = r0 + 2;
              acc2 +=
                  in_qm1 * float(weight[w_base + (uint)((stride - 1) - r2)]);
              acc2 += in_q0 *
                      float(weight[w_base + (uint)((kernel_size - 1) - r2)]);
            }
            if (valid_count > 3) {
              const int r3 = r0 + 3;
              acc3 +=
                  in_qm1 * float(weight[w_base + (uint)((stride - 1) - r3)]);
              acc3 += in_q0 *
                      float(weight[w_base + (uint)((kernel_size - 1) - r3)]);
            }
          } else {
            // stride < 4: lanes may have different q values.
            if (valid_count > 0) {
              if (q0 > 0 && q0 - 1 < seq_len_in) {
                const uint in_idx =
                    (b * (uint)seq_len_in + (uint)(q0 - 1)) * (uint)cin +
                    (uint)ic;
                acc0 += float(input[in_idx]) *
                        float(weight[w_base + (uint)((stride - 1) - r0)]);
              }
              if (q0 >= 0 && q0 < seq_len_in) {
                const uint in_idx =
                    (b * (uint)seq_len_in + (uint)q0) * (uint)cin + (uint)ic;
                acc0 += float(input[in_idx]) *
                        float(weight[w_base + (uint)((kernel_size - 1) - r0)]);
              }
            }
            if (valid_count > 1) {
              const int q1 = (t0 + 1) / stride;
              const int r1 = (t0 + 1) % stride;
              if (q1 > 0 && q1 - 1 < seq_len_in) {
                const uint in_idx =
                    (b * (uint)seq_len_in + (uint)(q1 - 1)) * (uint)cin +
                    (uint)ic;
                acc1 += float(input[in_idx]) *
                        float(weight[w_base + (uint)((stride - 1) - r1)]);
              }
              if (q1 >= 0 && q1 < seq_len_in) {
                const uint in_idx =
                    (b * (uint)seq_len_in + (uint)q1) * (uint)cin + (uint)ic;
                acc1 += float(input[in_idx]) *
                        float(weight[w_base + (uint)((kernel_size - 1) - r1)]);
              }
            }
            if (valid_count > 2) {
              const int q2 = (t0 + 2) / stride;
              const int r2 = (t0 + 2) % stride;
              if (q2 > 0 && q2 - 1 < seq_len_in) {
                const uint in_idx =
                    (b * (uint)seq_len_in + (uint)(q2 - 1)) * (uint)cin +
                    (uint)ic;
                acc2 += float(input[in_idx]) *
                        float(weight[w_base + (uint)((stride - 1) - r2)]);
              }
              if (q2 >= 0 && q2 < seq_len_in) {
                const uint in_idx =
                    (b * (uint)seq_len_in + (uint)q2) * (uint)cin + (uint)ic;
                acc2 += float(input[in_idx]) *
                        float(weight[w_base + (uint)((kernel_size - 1) - r2)]);
              }
            }
            if (valid_count > 3) {
              const int q3 = (t0 + 3) / stride;
              const int r3 = (t0 + 3) % stride;
              if (q3 > 0 && q3 - 1 < seq_len_in) {
                const uint in_idx =
                    (b * (uint)seq_len_in + (uint)(q3 - 1)) * (uint)cin +
                    (uint)ic;
                acc3 += float(input[in_idx]) *
                        float(weight[w_base + (uint)((stride - 1) - r3)]);
              }
              if (q3 >= 0 && q3 < seq_len_in) {
                const uint in_idx =
                    (b * (uint)seq_len_in + (uint)q3) * (uint)cin + (uint)ic;
                acc3 += float(input[in_idx]) *
                        float(weight[w_base + (uint)((kernel_size - 1) - r3)]);
              }
            }
          }
        } else {
          for (int k = 0; k < kernel_size; ++k) {
            const float w = float(weight[w_base + (uint)k]);
            // Lane 0
            if (valid_count > 0) {
              const int expanded_time0 = (int)t_out + k - left_pad;
              if (expanded_time0 >= 0 && expanded_time0 < seq_len_expanded &&
                  (expanded_time0 % stride) == 0) {
                const int src_time0 = expanded_time0 / stride;
                if (src_time0 >= 0 && src_time0 < seq_len_in) {
                  const uint in_idx =
                      (b * (uint)seq_len_in + (uint)src_time0) * (uint)cin +
                      (uint)ic;
                  acc0 += float(input[in_idx]) * w;
                }
              }
            }
            // Lane 1
            if (valid_count > 1) {
              const int expanded_time1 = (int)t_out + 1 + k - left_pad;
              if (expanded_time1 >= 0 && expanded_time1 < seq_len_expanded &&
                  (expanded_time1 % stride) == 0) {
                const int src_time1 = expanded_time1 / stride;
                if (src_time1 >= 0 && src_time1 < seq_len_in) {
                  const uint in_idx =
                      (b * (uint)seq_len_in + (uint)src_time1) * (uint)cin +
                      (uint)ic;
                  acc1 += float(input[in_idx]) * w;
                }
              }
            }
            // Lane 2
            if (valid_count > 2) {
              const int expanded_time2 = (int)t_out + 2 + k - left_pad;
              if (expanded_time2 >= 0 && expanded_time2 < seq_len_expanded &&
                  (expanded_time2 % stride) == 0) {
                const int src_time2 = expanded_time2 / stride;
                if (src_time2 >= 0 && src_time2 < seq_len_in) {
                  const uint in_idx =
                      (b * (uint)seq_len_in + (uint)src_time2) * (uint)cin +
                      (uint)ic;
                  acc2 += float(input[in_idx]) * w;
                }
              }
            }
            // Lane 3
            if (valid_count > 3) {
              const int expanded_time3 = (int)t_out + 3 + k - left_pad;
              if (expanded_time3 >= 0 && expanded_time3 < seq_len_expanded &&
                  (expanded_time3 % stride) == 0) {
                const int src_time3 = expanded_time3 / stride;
                if (src_time3 >= 0 && src_time3 < seq_len_in) {
                  const uint in_idx =
                      (b * (uint)seq_len_in + (uint)src_time3) * (uint)cin +
                      (uint)ic;
                  acc3 += float(input[in_idx]) * w;
                }
              }
            }
          }
        }
      }
    }
  }

  if (lane_count > 0) {
    output[out_base] = valid_count > 0 ? (T)acc0 : (T)0;
  }
  if (lane_count > 1) {
    output[out_base + 1] = valid_count > 1 ? (T)acc1 : (T)0;
  }
  if (lane_count > 2) {
    output[out_base + 2] = valid_count > 2 ? (T)acc2 : (T)0;
  }
  if (lane_count > 3) {
    output[out_base + 3] = valid_count > 3 ? (T)acc3 : (T)0;
  }
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
    uint t_out AXIS((seq_len_out + 3) / 4, 32),
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
