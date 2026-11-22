#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"

using namespace metal;

#define DEPTH_ATTENTION_THREADS 512u
#define DEPTH_ATTENTION_SIMDGROUPS (DEPTH_ATTENTION_THREADS / 32u)
// Channels per thread, loaded as two 4-wide vectors.
#define DEPTH_ATTENTION_VECTOR 8u
#define DEPTH_ATTENTION_MAX_MODEL_DIM (DEPTH_ATTENTION_THREADS * DEPTH_ATTENTION_VECTOR)
#define DEPTH_ATTENTION_MAX_SOURCES 8u
#define DEPTH_ATTENTION_MAX_QUERIES 4u
// Per source: the sum of squares, then one dot product per query.
#define DEPTH_ATTENTION_SUMS_PER_SOURCE (DEPTH_ATTENTION_MAX_QUERIES + 1u)
#define DEPTH_ATTENTION_SUMS (DEPTH_ATTENTION_MAX_SOURCES * DEPTH_ATTENTION_SUMS_PER_SOURCE)
static_assert(DEPTH_ATTENTION_SIMDGROUPS <= METAL_SIMD_SIZE, "one lane per simdgroup in the second reduction");

// Softmax mixes of RMS-normalized sources with learned queries, one threadgroup per token.
//
// The sources are rows of a slot-major stream [slot][stream_token][channel]; `source_slots` lists the slots to read,
// and the kernel reads `token_count` rows from `token_offset` on. Each query q yields
// output[q][token] = sum_s softmax_s(q . n_s / sqrt(model_dim)) * n_s, where n_s is source s scaled to unit RMS.
// Each thread loads its channels of every source once and keeps them in registers for the mix, so a token costs one
// round trip to memory. model_dim is a multiple of DEPTH_ATTENTION_VECTOR and at most DEPTH_ATTENTION_MAX_MODEL_DIM.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(DepthAttention)(
    device const T* stream,
    device const uint* source_slots,
    device const float* queries,
    device T* output,
    constant uint& token_count,
    constant uint& stream_token_count,
    constant uint& token_offset,
    constant uint& source_count,
    constant float& epsilon,
    const uint model_dim SPECIALIZE,
    const uint query_count SPECIALIZE,
    threadgroup float simdgroup_sums[DEPTH_ATTENTION_SUMS][DEPTH_ATTENTION_SIMDGROUPS],
    const ThreadContext thread_context,
    const uint token GROUPS(token_count),
    const uint thread_index THREADS(DEPTH_ATTENTION_THREADS)
) {
  using Vector = vec<T, 4>;
  const uint slot_stride = stream_token_count * model_dim;
  const uint row_offset = (token_offset + token) * model_dim;
  const uint channel = thread_index * DEPTH_ATTENTION_VECTOR;
  const bool active = channel < model_dim;
  const uint lane = thread_context.simd_lane_id;
  const uint simdgroup = thread_context.simdgroup_index;

  // All loads first: they are independent, so they overlap in one round trip.
  Vector values[DEPTH_ATTENTION_MAX_SOURCES][2];
  for (uint source = 0; source < DEPTH_ATTENTION_MAX_SOURCES; ++source) {
    if (active && source < source_count) {
      const device Vector* row =
          reinterpret_cast<const device Vector*>(stream + source_slots[source] * slot_stride + row_offset + channel);
      values[source][0] = row[0];
      values[source][1] = row[1];
    } else {
      values[source][0] = Vector(0);
      values[source][1] = Vector(0);
    }
  }

  for (uint source = 0; source < DEPTH_ATTENTION_MAX_SOURCES; ++source) {
    if (source >= source_count) {
      break;
    }
    const float4 low = float4(values[source][0]);
    const float4 high = float4(values[source][1]);
    const float squares = simd_sum(dot(low, low) + dot(high, high));
    if (lane == 0) {
      simdgroup_sums[source * DEPTH_ATTENTION_SUMS_PER_SOURCE][simdgroup] = squares;
    }
  }
  for (uint query = 0; query < DEPTH_ATTENTION_MAX_QUERIES; ++query) {
    if (query >= query_count) {
      break;
    }
    float4 query_low = float4(0.0f);
    float4 query_high = float4(0.0f);
    if (active) {
      const device float4* query_row = reinterpret_cast<const device float4*>(queries + query * model_dim + channel);
      query_low = query_row[0];
      query_high = query_row[1];
    }
    for (uint source = 0; source < DEPTH_ATTENTION_MAX_SOURCES; ++source) {
      if (source >= source_count) {
        break;
      }
      const float product =
          simd_sum(dot(float4(values[source][0]), query_low) + dot(float4(values[source][1]), query_high));
      if (lane == 0) {
        simdgroup_sums[source * DEPTH_ATTENTION_SUMS_PER_SOURCE + 1u + query][simdgroup] = product;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Simdgroup g totals the sums g, g + SIMDGROUPS, ..., its lanes reading one simdgroup's part each.
  const uint sum_count = source_count * DEPTH_ATTENTION_SUMS_PER_SOURCE;
  for (uint index = simdgroup; index < sum_count; index += DEPTH_ATTENTION_SIMDGROUPS) {
    if (index % DEPTH_ATTENTION_SUMS_PER_SOURCE > query_count) {
      continue;
    }
    const float total = simd_sum(lane < DEPTH_ATTENTION_SIMDGROUPS ? simdgroup_sums[index][lane] : 0.0f);
    if (lane == 0) {
      simdgroup_sums[index][0] = total;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float inverse_rms[DEPTH_ATTENTION_MAX_SOURCES];
  for (uint source = 0; source < DEPTH_ATTENTION_MAX_SOURCES; ++source) {
    inverse_rms[source] = source < source_count
        ? rsqrt(simdgroup_sums[source * DEPTH_ATTENTION_SUMS_PER_SOURCE][0] / float(model_dim) + epsilon)
        : 0.0f;
  }
  const float logit_scale = rsqrt(float(model_dim));
  for (uint query = 0; query < DEPTH_ATTENTION_MAX_QUERIES; ++query) {
    if (query >= query_count) {
      break;
    }
    // weights[source] multiplies the raw source values, so it includes the source's inverse RMS.
    float weights[DEPTH_ATTENTION_MAX_SOURCES];
    float maximum = -INFINITY;
    for (uint source = 0; source < DEPTH_ATTENTION_MAX_SOURCES; ++source) {
      if (source < source_count) {
        weights[source] = simdgroup_sums[source * DEPTH_ATTENTION_SUMS_PER_SOURCE + 1u + query][0] *
                          inverse_rms[source] * logit_scale;
        maximum = max(maximum, weights[source]);
      }
    }
    float denominator = 0.0f;
    for (uint source = 0; source < DEPTH_ATTENTION_MAX_SOURCES; ++source) {
      if (source < source_count) {
        weights[source] = exp(weights[source] - maximum);
        denominator += weights[source];
      }
    }
    float4 mixed_low = float4(0.0f);
    float4 mixed_high = float4(0.0f);
    for (uint source = 0; source < DEPTH_ATTENTION_MAX_SOURCES; ++source) {
      if (source < source_count) {
        const float weight = weights[source] / denominator * inverse_rms[source];
        mixed_low += weight * float4(values[source][0]);
        mixed_high += weight * float4(values[source][1]);
      }
    }
    if (active) {
      device Vector* out =
          reinterpret_cast<device Vector*>(output + (query * token_count + token) * model_dim + channel);
      out[0] = Vector(mixed_low);
      out[1] = Vector(mixed_high);
    }
  }
}

#define MESSAGE_GATE_THREADS 256u
#define MESSAGE_GATE_SIMDGROUPS (MESSAGE_GATE_THREADS / 32u)
#define MESSAGE_GATE_MAX_RANK 128u

// The low-rank projections z = down · [h / rms(h); m / rms(m)] of message gates, for every gate slot at once. Inputs
// are `input_slot_stride` elements apart, zero when the slots share one input. One threadgroup per token, slot and row:
// every thread loads its vectors of 8 channels of the row and of [h; m] at once, so a row costs one round trip, and the
// threadgroup also sums the squares of h and m for their RMS. model_dim is a multiple of 8 and at most
// MESSAGE_GATE_DOWN_MAX_MODEL_DIM.
#define MESSAGE_GATE_DOWN_VECTORS_PER_THREAD 4u
#define MESSAGE_GATE_DOWN_MAX_MODEL_DIM (MESSAGE_GATE_THREADS * MESSAGE_GATE_DOWN_VECTORS_PER_THREAD * 8u / 2u)
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MessageGateDown)(
    device const T* inputs,
    device const T* messages,
    device const bfloat* down_weights,
    device float* low_rank,
    constant uint& token_count,
    constant uint& input_slot_stride,
    constant float& epsilon,
    const uint model_dim SPECIALIZE,
    const uint rank SPECIALIZE,
    const uint slot_count SPECIALIZE,
    threadgroup float simdgroup_sums[4][MESSAGE_GATE_SIMDGROUPS],
    const ThreadContext thread_context,
    const uint token GROUPS(token_count),
    const uint slot GROUPS(slot_count),
    const uint row GROUPS(rank),
    const uint thread_index THREADS(MESSAGE_GATE_THREADS)
) {
  using Vector = vec<T, 4>;
  const device T* input_row = inputs + size_t(slot) * input_slot_stride + size_t(token) * model_dim;
  const device T* message_row = messages + (size_t(slot) * token_count + token) * model_dim;
  const device bfloat* weights = down_weights + (size_t(slot) * rank + row) * 2u * model_dim;
  const uint half_vectors = model_dim / 8u;
  float input_dot = 0.0f;
  float message_dot = 0.0f;
  float input_squares = 0.0f;
  float message_squares = 0.0f;
  METAL_PRAGMA_UNROLL
  for (uint step = 0; step < MESSAGE_GATE_DOWN_VECTORS_PER_THREAD; ++step) {
    const uint vector = thread_index + step * MESSAGE_GATE_THREADS;
    if (vector < 2u * half_vectors) {
      const bool is_message = vector >= half_vectors;
      const uint channel = (is_message ? vector - half_vectors : vector) * 8u;
      const device Vector* values = reinterpret_cast<const device Vector*>((is_message ? message_row : input_row) + channel);
      const device bfloat4* row_weights = reinterpret_cast<const device bfloat4*>(weights + vector * 8u);
      const float4 low = float4(values[0]);
      const float4 high = float4(values[1]);
      const float product = dot(float4(row_weights[0]), low) + dot(float4(row_weights[1]), high);
      const float squares = dot(low, low) + dot(high, high);
      input_dot += is_message ? 0.0f : product;
      message_dot += is_message ? product : 0.0f;
      input_squares += is_message ? 0.0f : squares;
      message_squares += is_message ? squares : 0.0f;
    }
  }
  const float sums[4] = {simd_sum(input_dot), simd_sum(message_dot), simd_sum(input_squares), simd_sum(message_squares)};
  if (thread_context.simd_lane_id == 0) {
    for (uint index = 0; index < 4u; ++index) {
      simdgroup_sums[index][thread_context.simdgroup_index] = sums[index];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index != 0) {
    return;
  }
  float totals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint simdgroup = 0; simdgroup < MESSAGE_GATE_SIMDGROUPS; ++simdgroup) {
    for (uint index = 0; index < 4u; ++index) {
      totals[index] += simdgroup_sums[index][simdgroup];
    }
  }
  const float input_scale = rsqrt(totals[2] / float(model_dim) + epsilon);
  const float message_scale = rsqrt(totals[3] / float(model_dim) + epsilon);
  low_rank[(size_t(slot) * token_count + token) * rank + row] = totals[0] * input_scale + totals[1] * message_scale;
}

// Mixes each slot's message into its input as (1 - tanh(F z)) * input + tanh(G z) * message, where z is the slot's
// low-rank projection and F, G are its float32 [model_dim, rank] matrices. Each simdgroup computes channels in turn,
// its lanes splitting the rank, so that the weight rows are read whole.
#define MESSAGE_GATE_CHANNELS_PER_SIMDGROUP 4u
#define MESSAGE_GATE_CHANNELS_PER_GROUP (MESSAGE_GATE_SIMDGROUPS * MESSAGE_GATE_CHANNELS_PER_SIMDGROUP)
#define MESSAGE_GATE_RANK_PER_LANE (MESSAGE_GATE_MAX_RANK / METAL_SIMD_SIZE)
static_assert(MESSAGE_GATE_SIMDGROUPS == 8u && MESSAGE_GATE_CHANNELS_PER_GROUP == 32u, "the GROUPS sizes above");

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MessageGateApply)(
    device const T* inputs,
    device const T* messages,
    device const float* low_rank,
    device const float* forget_weights,
    device const float* gain_weights,
    device T* output,
    constant uint& token_count,
    constant uint& input_slot_stride,
    const uint model_dim SPECIALIZE,
    const uint rank SPECIALIZE,
    const uint slot_count SPECIALIZE,
    const ThreadContext thread_context,
    const uint token GROUPS(token_count),
    const uint slot GROUPS(slot_count),
    const uint channel_block GROUPS(model_dim.div_ceil(32)), // MESSAGE_GATE_CHANNELS_PER_GROUP
    const uint thread_index THREADS(MESSAGE_GATE_THREADS)
) {
  const uint lane = thread_context.simd_lane_id;
  const size_t row = size_t(slot) * token_count + token;
  float z[MESSAGE_GATE_RANK_PER_LANE];
  for (uint i = 0; i < MESSAGE_GATE_RANK_PER_LANE; ++i) {
    const uint index = i * METAL_SIMD_SIZE + lane;
    z[i] = index < rank ? low_rank[row * rank + index] : 0.0f;
  }
  const device float* slot_forget = forget_weights + size_t(slot) * model_dim * rank;
  const device float* slot_gain = gain_weights + size_t(slot) * model_dim * rank;
  const device T* input_row = inputs + size_t(slot) * input_slot_stride + size_t(token) * model_dim;
  const uint first_channel = channel_block * MESSAGE_GATE_CHANNELS_PER_GROUP +
                             thread_context.simdgroup_index * MESSAGE_GATE_CHANNELS_PER_SIMDGROUP;
  for (uint channel = first_channel; channel < min(first_channel + MESSAGE_GATE_CHANNELS_PER_SIMDGROUP, model_dim);
       ++channel) {
    const device float* forget_row = slot_forget + channel * rank;
    const device float* gain_row = slot_gain + channel * rank;
    float forget = 0.0f;
    float gain = 0.0f;
    for (uint i = 0; i < MESSAGE_GATE_RANK_PER_LANE; ++i) {
      const uint index = i * METAL_SIMD_SIZE + lane;
      if (index < rank) {
        forget += forget_row[index] * z[i];
        gain += gain_row[index] * z[i];
      }
    }
    forget = simd_sum(forget);
    gain = simd_sum(gain);
    if (lane == 0) {
      const size_t offset = row * model_dim + channel;
      output[offset] = T((1.0f - tanh(forget)) * float(input_row[channel]) + tanh(gain) * float(messages[offset]));
    }
  }
}
