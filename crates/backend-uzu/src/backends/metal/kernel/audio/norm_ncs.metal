#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"

using namespace metal;

#define AUDIO_NORM_NCS_BLOCK_SIZE 256
#define AUDIO_NORM_NCS_MAX_SIMDS 32

template <typename T>
void norm_ncs(
    device const T* input,     // [B, C, T]
    device const T* scales,    // [C]
    device const T* bias,      // [C]
    device T* output,          // [B, C, T]
    device const int* lengths, // [B]
    const constant int& channels,
    const constant int& seq_len,
    const constant float& epsilon,
    const constant int& subtract_mean,
    threadgroup float* shared_mean,
    threadgroup float* shared_variance,
    const thread ThreadContext& thread_context,
    const uint b,
    const uint t,
    const uint lid
) {
  if (t >= (uint)seq_len || channels <= 0) {
    return;
  }

  const int len_b = lengths ? lengths[b] : seq_len;

  if ((int)t >= len_b) {
    for (uint c = lid; c < (uint)channels; c += AUDIO_NORM_NCS_BLOCK_SIZE) {
      const uint out_idx = (b * (uint)channels + c) * (uint)seq_len + t;
      output[out_idx] = (T)0;
    }
    return;
  }

  float partial_sum = 0.0f;
  if (subtract_mean != 0) {
    for (uint c = lid; c < (uint)channels; c += AUDIO_NORM_NCS_BLOCK_SIZE) {
      const uint idx = (b * (uint)channels + c) * (uint)seq_len + t;
      partial_sum += float(input[idx]);
    }
  }

  const float sum = threadgroup_cooperative_reduce<
      SimdReduceSum<float>,
      AUDIO_NORM_NCS_BLOCK_SIZE>(partial_sum, shared_mean, thread_context);
  const float mean = (subtract_mean != 0) ? (sum / (float)channels) : 0.0f;

  float partial_variance = 0.0f;
  for (uint c = lid; c < (uint)channels; c += AUDIO_NORM_NCS_BLOCK_SIZE) {
    const uint idx = (b * (uint)channels + c) * (uint)seq_len + t;
    const float x = float(input[idx]);
    const float centered = (subtract_mean != 0) ? (x - mean) : x;
    partial_variance += centered * centered;
  }

  const float variance_sum = threadgroup_cooperative_reduce<
      SimdReduceSum<float>,
      AUDIO_NORM_NCS_BLOCK_SIZE>(
      partial_variance,
      shared_variance,
      thread_context
  );
  const float inv_std = rsqrt(variance_sum / (float)channels + epsilon);

  for (uint c = lid; c < (uint)channels; c += AUDIO_NORM_NCS_BLOCK_SIZE) {
    const uint idx = (b * (uint)channels + c) * (uint)seq_len + t;
    const float x = float(input[idx]);
    const float centered = (subtract_mean != 0) ? (x - mean) : x;
    const float y = centered * inv_std * float(scales[c]) + float(bias[c]);
    output[idx] = (T)y;
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(AudioNormNcs)(
    device const T* input,
    device const T* scales,
    device const T* bias,
    device T* output,
    device const int* lengths,
    const constant int& channels,
    const constant int& seq_len,
    const constant float& epsilon,
    const constant int& subtract_mean,
    const constant int& batch_size,
    threadgroup float shared_mean[AUDIO_NORM_NCS_MAX_SIMDS],
    threadgroup float shared_variance[AUDIO_NORM_NCS_MAX_SIMDS],
    const ThreadContext thread_context,
    const uint b GROUPS(batch_size),
    const uint t GROUPS(seq_len),
    const uint lid THREADS(AUDIO_NORM_NCS_BLOCK_SIZE)
) {
  norm_ncs<T>(
      input,
      scales,
      bias,
      output,
      lengths,
      channels,
      seq_len,
      epsilon,
      subtract_mean,
      shared_mean,
      shared_variance,
      thread_context,
      b,
      t,
      lid
  );
}
