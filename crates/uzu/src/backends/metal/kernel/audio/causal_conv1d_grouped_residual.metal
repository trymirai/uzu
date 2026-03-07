#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

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
    output[out_idx] = residual[out_idx];
    return;
  }

  const int cin_per_group = cin / groups;
  const int cout_per_group = cout / groups;
  const int group_idx = (int)oc / cout_per_group;
  const int oc_in_group = (int)oc - group_idx * cout_per_group;
  const int in_begin = group_idx * cin_per_group;

  const int pad = (kernel_size - 1) * dilation;
  float acc = float(bias[oc]) + float(residual[out_idx]);

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

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioCausalConv1dGroupedResidual)(
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
    uint t AXIS(seq_len, 32),
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
