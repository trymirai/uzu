#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
void transpose_nsc_to_ncs(
    device const T* input, // [B, S, C]
    device T* output,      // [B, C, S]
    const constant int& seq_len,
    const constant int& channels,
    const uint3 gid
) {
  const uint t = gid.x;
  const uint c = gid.y;
  const uint b = gid.z;
  if (t >= (uint)seq_len || c >= (uint)channels) {
    return;
  }

  const uint src_idx = (b * (uint)seq_len + t) * (uint)channels + c;
  const uint dst_idx = (b * (uint)channels + c) * (uint)seq_len + t;
  output[dst_idx] = input[src_idx];
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioTransposeNscToNcs)(
    device const T* input,
    device T* output,
    const constant int& seq_len,
    const constant int& channels,
    const constant int& batch_size,
    uint t AXIS(seq_len, 32),
    uint c AXIS(channels, 1),
    uint b AXIS(batch_size, 1)
) {
  transpose_nsc_to_ncs<T>(
      input,
      output,
      seq_len,
      channels,
      uint3(t, c, b)
  );
}
