#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

template <typename T>
void conv1d(
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

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(AudioConv1d)(
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
  conv1d<T>(
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
