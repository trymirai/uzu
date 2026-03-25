#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

template <typename T>
void causal_conv_transpose1d(
    device const T* input,     // [B, Cin, Tin]
    device const T* weight,    // [Cout, Cin_per_group, 2*stride]
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

  const int q = (int)t_out / stride;
  const int r = (int)t_out - q * stride;

  float acc = float(bias[oc]);

  const uint in_base_b = b * (uint)cin * (uint)seq_len_in;
  const int ic_begin = group_idx * cin_per_group;
  const int ic_end = ic_begin + cin_per_group;

  for (int ic = ic_begin; ic < ic_end; ++ic) {
    if (q >= seq_len_in) {
      continue;
    }
    const uint in_base = in_base_b + (uint)ic * (uint)seq_len_in;

    // Weight layout: [Cout, Cin_per_group, K]
    const uint ic_local = (uint)(ic - ic_begin);
    const uint w_base =
        ((uint)oc * (uint)cin_per_group + ic_local) * (uint)(2 * stride);

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

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(AudioCausalConvTranspose1d)(
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
  causal_conv_transpose1d<T>(
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
