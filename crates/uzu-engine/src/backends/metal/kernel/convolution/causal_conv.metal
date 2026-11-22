#include <metal_stdlib>
#include "../common/dsl.h"
#include "../generated/trie.h"

using namespace metal;

#define CAUSAL_CONV_MAX_KERNEL_SIZE 8u

// Stateful depthwise causal convolution, computed in float32.
//
// `weights[channel * kernel_size + kernel_size - 1]` weights the current token. Dynamic taps add per-token deltas
// shared by contiguous groups of `group_size` channels, stored per token as [group][tap] in the weights' tap order.
// The identity adds the input before `output_scale`. `state` holds the previous kernel_size - 1 inputs of each
// channel, oldest first. A flat batch continues from `state` and rewrites it; a trie starts its roots from `state`
// and writes the inputs that each node leaves to `node_states`, so that accepting a node can copy them.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(CausalConv)(
    device const T* input,
    device const T* taps OPTIONAL(has_taps),
    device const float* weights,
    device float* state,
    device const uzu::trie::TrieNode* trie OPTIONAL(is_trie),
    device float* node_states OPTIONAL(is_trie),
    device T* output,
    constant uint& token_count,
    constant uint& input_stride,
    constant uint& taps_stride,
    constant float& output_scale,
    constant uint& accumulate,
    const uint channels SPECIALIZE,
    const uint kernel_size SPECIALIZE,
    const uint group_size SPECIALIZE,
    const bool has_identity SPECIALIZE,
    const bool has_taps SPECIALIZE,
    const bool is_trie SPECIALIZE,
    const uint channel AXIS(channels, 256)
) {
  const uint history = kernel_size - 1u;
  const device float* channel_weights = weights + channel * kernel_size;
  const uint tap_offset = (channel / group_size) * kernel_size;
  device float* channel_state = state + channel * history;

  float window[CAUSAL_CONV_MAX_KERNEL_SIZE];
  if (!is_trie) {
    for (uint tap = 0; tap < history; ++tap) {
      window[tap] = channel_state[tap];
    }
  }

  for (uint token = 0; token < token_count; ++token) {
    if (is_trie) {
      const int parent = trie[token].parent;
      const device float* parent_inputs =
          parent < 0 ? channel_state : node_states + (uint(parent) * channels + channel) * history;
      for (uint tap = 0; tap < history; ++tap) {
        window[tap] = parent_inputs[tap];
      }
    }

    const float value = float(input[token * input_stride + channel]);
    window[history] = value;
    float accumulator = 0.0f;
    for (uint tap = 0; tap < kernel_size; ++tap) {
      float weight = channel_weights[tap];
      if (has_taps) {
        weight += float(taps[token * taps_stride + tap_offset + tap]);
      }
      accumulator += weight * window[tap];
    }
    if (has_identity) {
      accumulator += value;
    }
    const uint output_index = token * channels + channel;
    output[output_index] = T((accumulate != 0 ? float(output[output_index]) : 0.0f) + accumulator * output_scale);

    for (uint tap = 0; tap < history; ++tap) {
      window[tap] = window[tap + 1u];
    }
    if (is_trie) {
      device float* inputs_left = node_states + (token * channels + channel) * history;
      for (uint tap = 0; tap < history; ++tap) {
        inputs_left[tap] = window[tap];
      }
    }
  }

  if (!is_trie) {
    for (uint tap = 0; tap < history; ++tap) {
      channel_state[tap] = window[tap];
    }
  }
}

// The flat causal convolution of many tokens, one thread per token and channel. It reads `state` without changing it:
// CausalConvShift then moves the last inputs into it.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(CausalConvTokens)(
    device const T* input,
    device const T* taps OPTIONAL(has_taps),
    device const float* weights,
    device const float* state,
    device T* output,
    constant uint& token_count,
    constant uint& input_stride,
    constant uint& taps_stride,
    constant float& output_scale,
    constant uint& accumulate,
    const uint channels SPECIALIZE,
    const uint kernel_size SPECIALIZE,
    const uint group_size SPECIALIZE,
    const bool has_identity SPECIALIZE,
    const bool has_taps SPECIALIZE,
    const uint token AXIS(token_count, 1),
    const uint channel AXIS(channels, 256)
) {
  const uint history = kernel_size - 1u;
  const device float* channel_weights = weights + channel * kernel_size;
  const uint tap_offset = token * taps_stride + (channel / group_size) * kernel_size;
  float accumulator = 0.0f;
  float value = 0.0f;
  for (uint tap = 0; tap < kernel_size; ++tap) {
    // Tap `kernel_size - 1` weights the current token; earlier taps reach back into the state.
    const int source = int(token + tap) - int(history);
    value = source >= 0 ? float(input[uint(source) * input_stride + channel])
                        : state[channel * history + uint(int(history) + source)];
    float weight = channel_weights[tap];
    if (has_taps) {
      weight += float(taps[tap_offset + tap]);
    }
    accumulator += weight * value;
  }
  if (has_identity) {
    accumulator += value;
  }
  const uint output_index = token * channels + channel;
  output[output_index] = T((accumulate != 0 ? float(output[output_index]) : 0.0f) + accumulator * output_scale);
}

// Moves the last kernel_size - 1 inputs of a flat batch, continuing from the old state, into the state.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(CausalConvShift)(
    device const T* input,
    device float* state,
    constant uint& token_count,
    constant uint& input_stride,
    const uint channels SPECIALIZE,
    const uint kernel_size SPECIALIZE,
    const uint channel AXIS(channels, 256)
) {
  const uint history = kernel_size - 1u;
  device float* channel_state = state + channel * history;
  float window[CAUSAL_CONV_MAX_KERNEL_SIZE];
  for (uint index = 0; index < history; ++index) {
    // The inputs follow the old state; the new state is the last `history` of both.
    const uint position = token_count + index;
    window[index] =
        position >= history ? float(input[(position - history) * input_stride + channel]) : channel_state[position];
  }
  for (uint index = 0; index < history; ++index) {
    channel_state[index] = window[index];
  }
}
