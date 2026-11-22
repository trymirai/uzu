#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"

using namespace metal;

#define TAP_MIX_THREADS 512u
#define TAP_MIX_SIMDGROUPS (TAP_MIX_THREADS / 32u)
// Channels per thread, loaded as two 4-wide vectors.
#define TAP_MIX_VECTOR 8u
#define TAP_MIX_TAPS 4u
#define TAP_MIX_MAX_MIXES 4u
#define MARKOV_THREADS 256u

// Softmax-weighted mixes of four tapped hidden states, each RMS-normalized with its own scales: for row r of
// `row_indices`, output[mix][r] = rms_norm(sum_t mix_weights[mix][t] * tap_t[row_indices[r]]) * norm_scales[mix].
// One threadgroup per row, every thread holding its 8 channels of the four taps; model_dim is a multiple of 8 and at
// most TAP_MIX_THREADS * TAP_MIX_VECTOR.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(TapMixNorm)(
    device const T* tap0,
    device const T* tap1,
    device const T* tap2,
    device const T* tap3,
    device const uint* row_indices,
    device const float* mix_weights,
    device const float* norm_scales,
    device T* output,
    constant uint& row_count,
    constant float& epsilon,
    const uint model_dim SPECIALIZE,
    const uint mix_count SPECIALIZE,
    threadgroup float simdgroup_sums[TAP_MIX_MAX_MIXES][TAP_MIX_SIMDGROUPS],
    const ThreadContext thread_context,
    const uint row GROUPS(row_count),
    const uint thread_index THREADS(TAP_MIX_THREADS)
) {
  using Vector = vec<T, 4>;
  const uint channel = thread_index * TAP_MIX_VECTOR;
  const bool active = channel < model_dim;
  const size_t source = size_t(row_indices[row]) * model_dim + channel;
  const device T* taps[TAP_MIX_TAPS] = {tap0, tap1, tap2, tap3};
  float4 values[TAP_MIX_TAPS][2];
  for (uint tap = 0; tap < TAP_MIX_TAPS; ++tap) {
    if (active) {
      const device Vector* tap_values = reinterpret_cast<const device Vector*>(taps[tap] + source);
      values[tap][0] = float4(tap_values[0]);
      values[tap][1] = float4(tap_values[1]);
    } else {
      values[tap][0] = float4(0.0f);
      values[tap][1] = float4(0.0f);
    }
  }

  float4 mixed[TAP_MIX_MAX_MIXES][2];
  for (uint mix = 0; mix < TAP_MIX_MAX_MIXES; ++mix) {
    if (mix >= mix_count) {
      break;
    }
    mixed[mix][0] = float4(0.0f);
    mixed[mix][1] = float4(0.0f);
    for (uint tap = 0; tap < TAP_MIX_TAPS; ++tap) {
      const float weight = mix_weights[mix * TAP_MIX_TAPS + tap];
      mixed[mix][0] += weight * values[tap][0];
      mixed[mix][1] += weight * values[tap][1];
    }
    const float squares = simd_sum(dot(mixed[mix][0], mixed[mix][0]) + dot(mixed[mix][1], mixed[mix][1]));
    if (thread_context.simd_lane_id == 0) {
      simdgroup_sums[mix][thread_context.simdgroup_index] = squares;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint mix = 0; mix < TAP_MIX_MAX_MIXES; ++mix) {
    if (mix >= mix_count) {
      break;
    }
    float total = 0.0f;
    for (uint simdgroup = 0; simdgroup < TAP_MIX_SIMDGROUPS; ++simdgroup) {
      total += simdgroup_sums[mix][simdgroup];
    }
    const float inverse_rms = rsqrt(total / float(model_dim) + epsilon);
    if (active) {
      const device float4* scales = reinterpret_cast<const device float4*>(norm_scales + mix * model_dim + channel);
      device Vector* out =
          reinterpret_cast<device Vector*>(output + (size_t(mix) * row_count + row) * model_dim + channel);
      out[0] = Vector(mixed[mix][0] * inverse_rms * scales[0]);
      out[1] = Vector(mixed[mix][1] * inverse_rms * scales[1]);
    }
  }
}

// Fills the inputs of a Markov head chain, [state, token embedding, draft hidden, context] per step: every step gets
// its draft hidden row (the rows after the anchor) and the context mix, and the first one a zero state and the
// anchor's embedding. The chain writes the later states and embeddings as it runs.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MarkovStart)(
    device const uint* anchor,
    device const T* token_embeddings,
    device const T* draft_hidden,
    device const T* markov_context,
    device T* inputs,
    constant uint& chain_length,
    const uint rank SPECIALIZE,
    const uint model_dim SPECIALIZE,
    const uint step GROUPS(chain_length),
    const uint thread_index THREADS(MARKOV_THREADS)
) {
  device T* row = inputs + size_t(step) * (2u * rank + 2u * model_dim);
  if (step == 0) {
    const size_t embedding = size_t(anchor[0]) * rank;
    for (uint index = thread_index; index < rank; index += MARKOV_THREADS) {
      row[index] = T(0);
      row[rank + index] = token_embeddings[embedding + index];
    }
  }
  const size_t hidden = size_t(step + 1u) * model_dim;
  for (uint index = thread_index; index < model_dim; index += MARKOV_THREADS) {
    row[2u * rank + index] = draft_hidden[hidden + index];
    row[2u * rank + model_dim + index] = markov_context[index];
  }
}

// One recurrence step of the Markov head: from projection = [gate, candidate, latent], the state becomes
// sigmoid(gate) * state + (1 - sigmoid(gate)) * tanh(candidate), written into the next step's inputs when there is a
// next step, and the latent is tanh(latent). Each product is rounded to T as in the reference.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MarkovStep)(
    device const T* projection,
    device const T* state,
    device T* next_state OPTIONAL(has_next),
    device T* latent,
    const uint rank SPECIALIZE,
    const bool has_next SPECIALIZE,
    const uint group GROUPS(rank.div_ceil(256)), // MARKOV_THREADS
    const uint thread_index THREADS(MARKOV_THREADS)
) {
  const uint index = group * MARKOV_THREADS + thread_index;
  if (index >= rank) {
    return;
  }
  const T gate = T(1.0f / (1.0f + exp(-float(projection[index]))));
  const T kept = T(float(gate) * float(state[index]));
  const T blend = T((1.0f - float(gate)) * float(T(tanh(float(projection[rank + index])))));
  if (has_next) {
    next_state[index] = T(float(kept) + float(blend));
  }
  latent[index] = T(tanh(float(projection[2u * rank + index])));
}

// The logits of a Markov head step: soft_cap(base + (bias + bias_offset)), where bias is the LM head folded with the
// head's output projection applied to the step's latent.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MarkovLogits)(
    device const T* base_logits,
    device const T* bias,
    device const T* bias_offset,
    device T* logits,
    constant uint& vocab_size,
    const constant float& soft_cap OPTIONAL(has_soft_cap),
    const bool has_soft_cap SPECIALIZE,
    const uint group GROUPS(vocab_size.div_ceil(256)), // MARKOV_THREADS
    const uint thread_index THREADS(MARKOV_THREADS)
) {
  const uint index = group * MARKOV_THREADS + thread_index;
  if (index >= vocab_size) {
    return;
  }
  float value = float(base_logits[index]) + float(T(float(bias[index]) + float(bias_offset[index])));
  if (has_soft_cap) {
    value = tanh(value / soft_cap) * soft_cap;
  }
  logits[index] = T(value);
}

// Writes the embedding of a step's sampled token into the next step's inputs.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MarkovEmbed)(
    device const uint* token,
    device const T* token_embeddings,
    device T* embedding,
    const uint rank SPECIALIZE,
    const uint group GROUPS(rank.div_ceil(256)), // MARKOV_THREADS
    const uint thread_index THREADS(MARKOV_THREADS)
) {
  const uint index = group * MARKOV_THREADS + thread_index;
  if (index < rank) {
    embedding[index] = token_embeddings[size_t(token[0]) * rank + index];
  }
}

static_assert(MARKOV_THREADS == 256u, "the GROUPS sizes above");
