#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
void causal_conv1d(
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
      uint3(t, oc, b)
  );
}
