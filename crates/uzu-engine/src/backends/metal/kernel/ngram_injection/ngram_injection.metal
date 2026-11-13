#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"

using namespace metal;

#define NGRAM_GATE_BLOCK_SIZE 256

PUBLIC KERNEL(NGramHash)(
    const device uint* tokens_with_history,
    const constant uint64_t* multipliers,
    const constant uint* head_vocab_sizes,
    const constant uint* head_offsets,
    device uint* ngram_ids,
    constant uint& num_tokens,
    constant uint& ngram_size,
    constant uint& heads_per_ngram,
    constant uint& eos_token_id,
    const uint head AXIS((ngram_size - 1) * heads_per_ngram, 32),
    const uint token AXIS(num_tokens, 1)
) {
  const uint num_heads = (ngram_size - 1) * heads_per_ngram;
  const uint position = ngram_size - 1 + token;
  const uint order = head / heads_per_ngram + 2;

  uint64_t hash = uint64_t(tokens_with_history[position]) * multipliers[0];
  bool crossed_eos = false;
  for (uint shift = 1; shift < order; ++shift) {
    const uint shifted_token = crossed_eos ? eos_token_id : tokens_with_history[position - shift];
    crossed_eos = crossed_eos || shifted_token == eos_token_id;
    hash ^= uint64_t(shifted_token) * multipliers[shift];
  }
  ngram_ids[token * num_heads + head] = uint(hash % uint64_t(head_vocab_sizes[head])) + head_offsets[head];
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(NGramGate)(
    const device T* keys,
    const device T* stream,
    const device T* values,
    const device float* key_scales,
    const device float* query_scales,
    const device float* conv_scales,
    device T* gated,
    device T* conv_input,
    constant uint& num_tokens,
    constant uint& stream_count,
    constant uint& model_dim,
    constant float& epsilon,
    constant float& scale_offset,
    threadgroup float shared[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint row GROUPS(num_tokens * stream_count),
    const uint thread_in_row THREADS(NGRAM_GATE_BLOCK_SIZE)
) {
  const uint token = row / stream_count;
  const uint stream_index = row % stream_count;
  const uint row_offset = row * model_dim;
  const uint value_offset = token * model_dim;
  const uint scale_row_offset = stream_index * model_dim;
  const float dim = float(model_dim);

  float key_square_sum = 0.0f;
  float query_square_sum = 0.0f;
  float scaled_dot = 0.0f;
  float value_square_sum = 0.0f;
  for (uint i = thread_in_row; i < model_dim; i += NGRAM_GATE_BLOCK_SIZE) {
    const float key = float(keys[row_offset + i]);
    const float query = float(stream[row_offset + i]);
    const float value = float(values[value_offset + i]);
    const float key_scale = key_scales[scale_row_offset + i] + scale_offset;
    const float query_scale = query_scales[scale_row_offset + i] + scale_offset;
    key_square_sum += key * key;
    query_square_sum += query * query;
    scaled_dot += key * key_scale * query * query_scale;
    value_square_sum += value * value;
  }

  key_square_sum = threadgroup_cooperative_reduce<SimdReduceSum<float>, NGRAM_GATE_BLOCK_SIZE>(
      key_square_sum,
      shared,
      thread_context
  );
  query_square_sum = threadgroup_cooperative_reduce<SimdReduceSum<float>, NGRAM_GATE_BLOCK_SIZE>(
      query_square_sum,
      shared,
      thread_context
  );
  scaled_dot =
      threadgroup_cooperative_reduce<SimdReduceSum<float>, NGRAM_GATE_BLOCK_SIZE>(scaled_dot, shared, thread_context);
  value_square_sum = threadgroup_cooperative_reduce<SimdReduceSum<float>, NGRAM_GATE_BLOCK_SIZE>(
      value_square_sum,
      shared,
      thread_context
  );

  const float key_rms_inv = rsqrt(key_square_sum / dim + epsilon);
  const float query_rms_inv = rsqrt(query_square_sum / dim + epsilon);
  const float logit = key_rms_inv * query_rms_inv * scaled_dot / sqrt(dim);
  const float signed_root = sign(logit) * sqrt(max(abs(logit), 1e-6f));
  const float gate = 1.0f / (1.0f + exp(-signed_root));
  const float conv_rms_inv = rsqrt(gate * gate * value_square_sum / dim + epsilon);

  for (uint i = thread_in_row; i < model_dim; i += NGRAM_GATE_BLOCK_SIZE) {
    const float gated_value = gate * float(values[value_offset + i]);
    const float conv_scale = conv_scales[scale_row_offset + i] + scale_offset;
    gated[row_offset + i] = T(gated_value);
    conv_input[row_offset + i] = T(gated_value * conv_rms_inv * conv_scale);
  }
}

template <typename T>
static float ngram_conv_sample(
    const device T* conv_input,
    const device T* state,
    uint padded_row,
    uint state_rows,
    uint channels,
    uint channel
) {
  return padded_row < state_rows ? float(state[padded_row * channels + channel])
                                 : float(conv_input[(padded_row - state_rows) * channels + channel]);
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(NGramConvInject)(
    const device T* conv_input,
    const device T* state,
    const device float* weights,
    const device T* gated,
    device T* stream,
    device T* next_state,
    constant uint& num_tokens,
    constant uint& channels,
    constant uint& kernel_size,
    const uint channel AXIS(channels, 256),
    const uint row AXIS(num_tokens + kernel_size - 1, 1)
) {
  const uint state_rows = kernel_size - 1;

  if (row < num_tokens) {
    float accumulator = 0.0f;
    for (uint tap = 0; tap < kernel_size; ++tap) {
      accumulator += weights[channel * kernel_size + tap] *
                     ngram_conv_sample(conv_input, state, row + tap, state_rows, channels, channel);
    }
    const float activated = accumulator / (1.0f + exp(-accumulator));
    const uint index = row * channels + channel;
    stream[index] = T(float(stream[index]) + float(gated[index]) + activated);
  } else {
    const uint state_row = row - num_tokens;
    next_state[state_row * channels + channel] =
        T(ngram_conv_sample(conv_input, state, num_tokens + state_row, state_rows, channels, channel));
  }
}
