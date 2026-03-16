#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

constant uint AUDIO_TIME_TILE = 4;

template <typename T>
void causal_conv1d_grouped_residual(
    device const T* input,     // [B, Cin, T]
    device const T* residual,  // [B, Cout, T]
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

  const float bias_value = float(bias[oc]);
  const float residual0 = lane_count > 0 ? float(residual[out_base]) : 0.0f;
  const float residual1 = lane_count > 1 ? float(residual[out_base + 1]) : 0.0f;
  const float residual2 = lane_count > 2 ? float(residual[out_base + 2]) : 0.0f;
  const float residual3 = lane_count > 3 ? float(residual[out_base + 3]) : 0.0f;
  float acc0 = bias_value + residual0;
  float acc1 = bias_value + residual1;
  float acc2 = bias_value + residual2;
  float acc3 = bias_value + residual3;

  const int cin_per_group = cin / groups;
  const int cout_per_group = cout / groups;
  const int group_idx = (int)oc / cout_per_group;
  const int oc_in_group = (int)oc - group_idx * cout_per_group;
  const int in_begin = group_idx * cin_per_group;

  const int pad = (kernel_size - 1) * dilation;
  if (valid_count > 0) {
    const bool full_tile = valid_count == (int)AUDIO_TIME_TILE;
    for (int ic_local = 0; ic_local < cin_per_group; ++ic_local) {
      const int ic = in_begin + ic_local;
      const uint w_base =
          ((group_idx * cout_per_group + oc_in_group) * (uint)cin_per_group +
           (uint)ic_local) *
          (uint)kernel_size;
      const uint x_base = (b * (uint)cin + (uint)ic) * (uint)seq_len;

      for (int k = 0; k < kernel_size; ++k) {
        const float w = float(weight[w_base + (uint)k]);
        const int x_t = (int)t + k * dilation - pad;
        if (full_tile && x_t >= 0 && x_t + 3 < seq_len) {
          const uint x_idx = x_base + (uint)x_t;
          acc0 += w * float(input[x_idx]);
          acc1 += w * float(input[x_idx + 1]);
          acc2 += w * float(input[x_idx + 2]);
          acc3 += w * float(input[x_idx + 3]);
          continue;
        }
        if (valid_count > 0 && x_t >= 0 && x_t < seq_len) {
          acc0 += w * float(input[x_base + (uint)x_t]);
        }
        if (valid_count > 1 && x_t + 1 >= 0 && x_t + 1 < seq_len) {
          acc1 += w * float(input[x_base + (uint)(x_t + 1)]);
        }
        if (valid_count > 2 && x_t + 2 >= 0 && x_t + 2 < seq_len) {
          acc2 += w * float(input[x_base + (uint)(x_t + 2)]);
        }
        if (valid_count > 3 && x_t + 3 >= 0 && x_t + 3 < seq_len) {
          acc3 += w * float(input[x_base + (uint)(x_t + 3)]);
        }
      }
    }
  }

  if (lane_count > 0) {
    output[out_base] = valid_count > 0 ? (T)acc0 : (T)residual0;
  }
  if (lane_count > 1) {
    output[out_base + 1] = valid_count > 1 ? (T)acc1 : (T)residual1;
  }
  if (lane_count > 2) {
    output[out_base + 2] = valid_count > 2 ? (T)acc2 : (T)residual2;
  }
  if (lane_count > 3) {
    output[out_base + 3] = valid_count > 3 ? (T)acc3 : (T)residual3;
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(AudioCausalConv1dGroupedResidual)(
    device const T* input,
    device const T* residual,
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
    uint t AXIS((seq_len + 3) / 4, 32),
    uint oc AXIS(cout, 1),
    uint b AXIS(batch_size, 1)
) {
  causal_conv1d_grouped_residual<T>(
      input,
      residual,
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
