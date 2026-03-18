#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

constant uint AUDIO_TIME_TILE = 8;
constant int AUDIO_LAYOUT_NCS = 0;
constant int AUDIO_LAYOUT_NSC = 1;

template <typename T>
void causal_conv1d_grouped_ncs(
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
  const uint t = gid.x * AUDIO_TIME_TILE;
  const uint oc = gid.y;
  const uint b = gid.z;

  if (t >= (uint)seq_len || oc >= (uint)cout) {
    return;
  }

  if (groups <= 0 || (cin % groups) != 0 || (cout % groups) != 0) {
    return;
  }

  const int len_b = lengths ? lengths[b] : seq_len;
  const uint out_base = (b * (uint)cout + oc) * (uint)seq_len + t;
  int lane_count = seq_len - (int)t;
  if (lane_count > (int)AUDIO_TIME_TILE) {
    lane_count = (int)AUDIO_TIME_TILE;
  }
  if (lane_count <= 0) {
    return;
  }
  int valid_count = len_b - (int)t;
  if (valid_count > lane_count) {
    valid_count = lane_count;
  }
  if (valid_count < 0) {
    valid_count = 0;
  }

  const int cin_per_group = cin / groups;
  const int cout_per_group = cout / groups;
  const int group_idx = (int)oc / cout_per_group;
  const int oc_in_group = (int)oc - group_idx * cout_per_group;
  const int in_begin = group_idx * cin_per_group;

  const int pad = (kernel_size - 1) * dilation;
  const float bias_value = float(bias[oc]);
  float acc0 = bias_value;
  float acc1 = bias_value;
  float acc2 = bias_value;
  float acc3 = bias_value;
  float acc4 = bias_value;
  float acc5 = bias_value;
  float acc6 = bias_value;
  float acc7 = bias_value;

  if (valid_count > 0) {
    const bool full_tile = valid_count == (int)AUDIO_TIME_TILE;
    for (int ic_local = 0; ic_local < cin_per_group; ++ic_local) {
      const uint w_base =
          ((group_idx * cout_per_group + oc_in_group) * (uint)cin_per_group +
           (uint)ic_local) *
          (uint)kernel_size;
      const int ic = in_begin + ic_local;
      const uint x_base_ncs = (b * (uint)cin + (uint)ic) * (uint)seq_len;

      for (int k = 0; k < kernel_size; ++k) {
        const float w = float(weight[w_base + (uint)k]);
        const int x_t = (int)t + k * dilation - pad;
        if (full_tile && x_t >= 0 && x_t + 7 < seq_len) {
          // Fast path: all 8 lanes in bounds, use vectorized loads
          const uint x_idx = x_base_ncs + (uint)x_t;
          vec<T, 4> v0 = *(device const vec<T, 4>*)(input + x_idx);
          vec<T, 4> v1 = *(device const vec<T, 4>*)(input + x_idx + 4);
          acc0 += w * float(v0.x);
          acc1 += w * float(v0.y);
          acc2 += w * float(v0.z);
          acc3 += w * float(v0.w);
          acc4 += w * float(v1.x);
          acc5 += w * float(v1.y);
          acc6 += w * float(v1.z);
          acc7 += w * float(v1.w);
          continue;
        }
        // Slow path: per-lane boundary checks
        if (valid_count > 0 && x_t >= 0 && x_t < seq_len) {
          acc0 += w * float(input[x_base_ncs + (uint)x_t]);
        }
        if (valid_count > 1 && x_t + 1 >= 0 && x_t + 1 < seq_len) {
          acc1 += w * float(input[x_base_ncs + (uint)(x_t + 1)]);
        }
        if (valid_count > 2 && x_t + 2 >= 0 && x_t + 2 < seq_len) {
          acc2 += w * float(input[x_base_ncs + (uint)(x_t + 2)]);
        }
        if (valid_count > 3 && x_t + 3 >= 0 && x_t + 3 < seq_len) {
          acc3 += w * float(input[x_base_ncs + (uint)(x_t + 3)]);
        }
        if (valid_count > 4 && x_t + 4 >= 0 && x_t + 4 < seq_len) {
          acc4 += w * float(input[x_base_ncs + (uint)(x_t + 4)]);
        }
        if (valid_count > 5 && x_t + 5 >= 0 && x_t + 5 < seq_len) {
          acc5 += w * float(input[x_base_ncs + (uint)(x_t + 5)]);
        }
        if (valid_count > 6 && x_t + 6 >= 0 && x_t + 6 < seq_len) {
          acc6 += w * float(input[x_base_ncs + (uint)(x_t + 6)]);
        }
        if (valid_count > 7 && x_t + 7 >= 0 && x_t + 7 < seq_len) {
          acc7 += w * float(input[x_base_ncs + (uint)(x_t + 7)]);
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
  if (lane_count > 4) {
    output[out_base + 4] = valid_count > 4 ? (T)acc4 : (T)0;
  }
  if (lane_count > 5) {
    output[out_base + 5] = valid_count > 5 ? (T)acc5 : (T)0;
  }
  if (lane_count > 6) {
    output[out_base + 6] = valid_count > 6 ? (T)acc6 : (T)0;
  }
  if (lane_count > 7) {
    output[out_base + 7] = valid_count > 7 ? (T)acc7 : (T)0;
  }
}

template <typename T>
void causal_conv1d_grouped_nsc(
    device const T* input,     // [B, T, Cin]
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
  const uint t = gid.x * AUDIO_TIME_TILE;
  const uint oc = gid.y;
  const uint b = gid.z;

  if (t >= (uint)seq_len || oc >= (uint)cout) {
    return;
  }

  if (groups <= 0 || (cin % groups) != 0 || (cout % groups) != 0) {
    return;
  }

  const int len_b = lengths ? lengths[b] : seq_len;
  const uint out_base = (b * (uint)cout + oc) * (uint)seq_len + t;
  int lane_count = seq_len - (int)t;
  if (lane_count > (int)AUDIO_TIME_TILE) {
    lane_count = (int)AUDIO_TIME_TILE;
  }
  if (lane_count <= 0) {
    return;
  }
  int valid_count = len_b - (int)t;
  if (valid_count > lane_count) {
    valid_count = lane_count;
  }
  if (valid_count < 0) {
    valid_count = 0;
  }

  const int cin_per_group = cin / groups;
  const int cout_per_group = cout / groups;
  const int group_idx = (int)oc / cout_per_group;
  const int oc_in_group = (int)oc - group_idx * cout_per_group;
  const int in_begin = group_idx * cin_per_group;

  const int pad = (kernel_size - 1) * dilation;
  const float bias_value = float(bias[oc]);
  float acc0 = bias_value;
  float acc1 = bias_value;
  float acc2 = bias_value;
  float acc3 = bias_value;
  float acc4 = bias_value;
  float acc5 = bias_value;
  float acc6 = bias_value;
  float acc7 = bias_value;

  if (valid_count > 0) {
    const bool full_tile = valid_count == (int)AUDIO_TIME_TILE;
    for (int ic_local = 0; ic_local < cin_per_group; ++ic_local) {
      const int ic = in_begin + ic_local;
      const uint w_base =
          ((group_idx * cout_per_group + oc_in_group) * (uint)cin_per_group +
           (uint)ic_local) *
          (uint)kernel_size;

      for (int k = 0; k < kernel_size; ++k) {
        const float w = float(weight[w_base + (uint)k]);
        const int x_t = (int)t + k * dilation - pad;
        if (full_tile && x_t >= 0 && x_t + 7 < seq_len) {
          const uint x_idx0 = (b * (uint)seq_len + (uint)x_t) * (uint)cin + (uint)ic;
          acc0 += w * float(input[x_idx0]);
          acc1 += w * float(input[x_idx0 + (uint)cin]);
          acc2 += w * float(input[x_idx0 + (uint)(2 * cin)]);
          acc3 += w * float(input[x_idx0 + (uint)(3 * cin)]);
          acc4 += w * float(input[x_idx0 + (uint)(4 * cin)]);
          acc5 += w * float(input[x_idx0 + (uint)(5 * cin)]);
          acc6 += w * float(input[x_idx0 + (uint)(6 * cin)]);
          acc7 += w * float(input[x_idx0 + (uint)(7 * cin)]);
          continue;
        }
        if (valid_count > 0 && x_t >= 0 && x_t < seq_len) {
          const uint x_idx = (b * (uint)seq_len + (uint)x_t) * (uint)cin + (uint)ic;
          acc0 += w * float(input[x_idx]);
        }
        if (valid_count > 1 && x_t + 1 >= 0 && x_t + 1 < seq_len) {
          const uint x_idx = (b * (uint)seq_len + (uint)(x_t + 1)) * (uint)cin + (uint)ic;
          acc1 += w * float(input[x_idx]);
        }
        if (valid_count > 2 && x_t + 2 >= 0 && x_t + 2 < seq_len) {
          const uint x_idx = (b * (uint)seq_len + (uint)(x_t + 2)) * (uint)cin + (uint)ic;
          acc2 += w * float(input[x_idx]);
        }
        if (valid_count > 3 && x_t + 3 >= 0 && x_t + 3 < seq_len) {
          const uint x_idx = (b * (uint)seq_len + (uint)(x_t + 3)) * (uint)cin + (uint)ic;
          acc3 += w * float(input[x_idx]);
        }
        if (valid_count > 4 && x_t + 4 >= 0 && x_t + 4 < seq_len) {
          const uint x_idx = (b * (uint)seq_len + (uint)(x_t + 4)) * (uint)cin + (uint)ic;
          acc4 += w * float(input[x_idx]);
        }
        if (valid_count > 5 && x_t + 5 >= 0 && x_t + 5 < seq_len) {
          const uint x_idx = (b * (uint)seq_len + (uint)(x_t + 5)) * (uint)cin + (uint)ic;
          acc5 += w * float(input[x_idx]);
        }
        if (valid_count > 6 && x_t + 6 >= 0 && x_t + 6 < seq_len) {
          const uint x_idx = (b * (uint)seq_len + (uint)(x_t + 6)) * (uint)cin + (uint)ic;
          acc6 += w * float(input[x_idx]);
        }
        if (valid_count > 7 && x_t + 7 >= 0 && x_t + 7 < seq_len) {
          const uint x_idx = (b * (uint)seq_len + (uint)(x_t + 7)) * (uint)cin + (uint)ic;
          acc7 += w * float(input[x_idx]);
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
  if (lane_count > 4) {
    output[out_base + 4] = valid_count > 4 ? (T)acc4 : (T)0;
  }
  if (lane_count > 5) {
    output[out_base + 5] = valid_count > 5 ? (T)acc5 : (T)0;
  }
  if (lane_count > 6) {
    output[out_base + 6] = valid_count > 6 ? (T)acc6 : (T)0;
  }
  if (lane_count > 7) {
    output[out_base + 7] = valid_count > 7 ? (T)acc7 : (T)0;
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(AudioCausalConv1dGrouped)(
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
    const constant int& input_layout,
    const constant int& batch_size,
    uint t AXIS((seq_len + 7) / 8, 32),
    uint oc AXIS(cout, 1),
    uint b AXIS(batch_size, 1)
) {
  if (input_layout == AUDIO_LAYOUT_NCS) {
    causal_conv1d_grouped_ncs<T>(
        input, weight, bias, output, lengths,
        cin, cout, seq_len, kernel_size, dilation, groups,
        uint3(t, oc, b)
    );
  } else {
    causal_conv1d_grouped_nsc<T>(
        input, weight, bias, output, lengths,
        cin, cout, seq_len, kernel_size, dilation, groups,
        uint3(t, oc, b)
    );
  }
}
