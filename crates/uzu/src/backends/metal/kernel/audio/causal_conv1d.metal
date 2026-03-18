#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

constant uint AUDIO_TIME_TILE = 4;
constant int AUDIO_LAYOUT_NCS = 0;
constant int AUDIO_LAYOUT_NSC = 1;

template <typename T>
void causal_conv1d(
    device const T* input,     // [B, Cin, T] when NCS, [B, T, Cin] when NSC
    device const T* weight,    // [Cout, Cin, K]
    device const T* bias,      // [Cout]
    device T* output,          // [B, Cout, T]
    device const int* lengths, // [B]
    const constant int& cin,
    const constant int& cout,
    const constant int& seq_len,
    const constant int& kernel_size,
    const constant int& dilation,
    const constant int& input_layout,
    const uint3 gid
) {
  const uint t = gid.x * AUDIO_TIME_TILE;
  const uint oc = gid.y;
  const uint b = gid.z;

  if (t >= (uint)seq_len || oc >= (uint)cout) {
    return;
  }

  if (input_layout != AUDIO_LAYOUT_NCS && input_layout != AUDIO_LAYOUT_NSC) {
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

  const int pad = (kernel_size - 1) * dilation;
  const float bias_value = float(bias[oc]);
  float acc0 = bias_value;
  float acc1 = bias_value;
  float acc2 = bias_value;
  float acc3 = bias_value;

  if (valid_count > 0) {
    const bool full_tile = valid_count == (int)AUDIO_TIME_TILE;
    for (int ic = 0; ic < cin; ++ic) {
      const uint w_base = (oc * (uint)cin + (uint)ic) * (uint)kernel_size;
      const uint x_base_ncs = (b * (uint)cin + (uint)ic) * (uint)seq_len;
      for (int k = 0; k < kernel_size; ++k) {
        const float w = float(weight[w_base + (uint)k]);
        const int x_t = (int)t + k * dilation - pad;
        if (full_tile && x_t >= 0 && x_t + 3 < seq_len) {
          if (input_layout == AUDIO_LAYOUT_NCS) {
            const uint x_idx = x_base_ncs + (uint)x_t;
            acc0 += w * float(input[x_idx]);
            acc1 += w * float(input[x_idx + 1]);
            acc2 += w * float(input[x_idx + 2]);
            acc3 += w * float(input[x_idx + 3]);
          } else {
            const uint x_idx0 = (b * (uint)seq_len + (uint)x_t) * (uint)cin + (uint)ic;
            acc0 += w * float(input[x_idx0]);
            acc1 += w * float(input[x_idx0 + (uint)cin]);
            acc2 += w * float(input[x_idx0 + (uint)(2 * cin)]);
            acc3 += w * float(input[x_idx0 + (uint)(3 * cin)]);
          }
          continue;
        }
        if (valid_count > 0 && x_t >= 0 && x_t < seq_len) {
          const uint x_idx = (input_layout == AUDIO_LAYOUT_NCS)
              ? (x_base_ncs + (uint)x_t)
              : ((b * (uint)seq_len + (uint)x_t) * (uint)cin + (uint)ic);
          acc0 += w * float(input[x_idx]);
        }
        if (valid_count > 1 && x_t + 1 >= 0 && x_t + 1 < seq_len) {
          const uint x_idx = (input_layout == AUDIO_LAYOUT_NCS)
              ? (x_base_ncs + (uint)(x_t + 1))
              : ((b * (uint)seq_len + (uint)(x_t + 1)) * (uint)cin + (uint)ic);
          acc1 += w * float(input[x_idx]);
        }
        if (valid_count > 2 && x_t + 2 >= 0 && x_t + 2 < seq_len) {
          const uint x_idx = (input_layout == AUDIO_LAYOUT_NCS)
              ? (x_base_ncs + (uint)(x_t + 2))
              : ((b * (uint)seq_len + (uint)(x_t + 2)) * (uint)cin + (uint)ic);
          acc2 += w * float(input[x_idx]);
        }
        if (valid_count > 3 && x_t + 3 >= 0 && x_t + 3 < seq_len) {
          const uint x_idx = (input_layout == AUDIO_LAYOUT_NCS)
              ? (x_base_ncs + (uint)(x_t + 3))
              : ((b * (uint)seq_len + (uint)(x_t + 3)) * (uint)cin + (uint)ic);
          acc3 += w * float(input[x_idx]);
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
PUBLIC KERNEL(AudioCausalConv1d)(
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
    const constant int& input_layout,
    const constant int& batch_size,
    uint t AXIS((seq_len + 3) / 4, 32),
    uint oc AXIS(cout, 1),
    uint b AXIS(batch_size, 1)
) {
  causal_conv1d<T>(
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
      input_layout,
      uint3(t, oc, b)
  );
}
