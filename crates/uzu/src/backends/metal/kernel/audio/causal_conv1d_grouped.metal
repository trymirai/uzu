#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

constant int AUDIO_LAYOUT_NCS = 0;
constant int AUDIO_LAYOUT_NSC = 1;

template <typename T>
void causal_conv1d_grouped(
    device const T* input,     // [B, Cin, T] when NCS, [B, T, Cin] when NSC
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
    const constant int& input_layout,
    const uint3 gid
) {
  const uint t = gid.x;
  const uint oc = gid.y;
  const uint b = gid.z;

  if (t >= (uint)seq_len || oc >= (uint)cout) {
    return;
  }

  if ((input_layout != AUDIO_LAYOUT_NCS && input_layout != AUDIO_LAYOUT_NSC) ||
      groups <= 0 || (cin % groups) != 0 || (cout % groups) != 0) {
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
    for (int k = 0; k < kernel_size; ++k) {
      const int x_t = (int)t + k * dilation - pad;
      if (x_t < 0 || x_t >= seq_len) {
        continue;
      }
      const uint x_idx = (input_layout == AUDIO_LAYOUT_NCS)
          ? (((b * (uint)cin + (uint)ic) * (uint)seq_len) + (uint)x_t)
          : (((b * (uint)seq_len + (uint)x_t) * (uint)cin) + (uint)ic);
      acc += float(weight[w_base + (uint)k]) * float(input[x_idx]);
    }
  }

  output[out_idx] = (T)acc;
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
    uint t AXIS(seq_len, 32),
    uint oc AXIS(cout, 1),
    uint b AXIS(batch_size, 1)
) {
  causal_conv1d_grouped<T>(
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
      input_layout,
      uint3(t, oc, b)
  );
}
