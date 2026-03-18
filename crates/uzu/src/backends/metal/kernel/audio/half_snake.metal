#include <metal_stdlib>
#include "../common/dsl.h"
#include "../activation/activation.h"

using namespace metal;

template <typename T>
void half_snake(
    device const T* input, // [B, C, T]
    device const T* alpha, // [1, C_snake, 1] (contiguous, index by channel)
    device T* output,      // [B, C, T]
    const constant int& channels,
    const constant int& seq_len,
    const constant int& snake_channels,
    const constant float& negative_slope,
    const constant float& eps,
    const uint3 gid
) {
  const uint t = gid.x;
  const uint c = gid.y;
  const uint b = gid.z;

  if (t >= (uint)seq_len || c >= (uint)channels) {
    return;
  }

  const uint idx = (b * (uint)channels + c) * (uint)seq_len + t;
  const float x = float(input[idx]);

  if ((int)c < snake_channels) {
    const float a = float(alpha[c]);
    const float ax = a * x;
    const float s = sin(ax);
    const float y = x + (1.0f / (a + eps)) * (s * s);
    output[idx] = (T)y;
  } else {
    output[idx] = activate((T)x, ACT_LEAKY_RELU, negative_slope);
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(AudioHalfSnake)(
    device const T* input,
    device const T* alpha,
    device T* output,
    const constant int& channels,
    const constant int& seq_len,
    const constant int& snake_channels,
    const constant float& negative_slope,
    const constant float& eps,
    const constant int& batch_size,
    uint t AXIS(seq_len, 32),
    uint c AXIS(channels, 1),
    uint b AXIS(batch_size, 1)
) {
  half_snake<T>(
      input,
      alpha,
      output,
      channels,
      seq_len,
      snake_channels,
      negative_slope,
      eps,
      uint3(t, c, b)
  );
}
