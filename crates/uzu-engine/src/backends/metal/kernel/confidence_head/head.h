#pragma once

#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/thread_context.h"
#include "../activation/activations.h"
#include "../generated/confidence_head.h"

using namespace metal;
using namespace uzu::confidence_head;

#define HEAD_THREADS 512u
#define HEAD_SIMDGROUPS (HEAD_THREADS / 32u)
#define HEAD_EDGE_INPUT_MAX (3u * MODEL_MAX + SLOT_NUMBERS)
#define HEAD_DENSE_UNITS 4u
#define HEAD_NONE 0xffffffffu
#define HEAD_LOG_PROB_FLOOR (-30.0f)
#define HEAD_RELATION_ABSENT (RELATIONS + 1u)

METAL_FUNC float head_log_sum_exp(float value, threadgroup float* reduce, const thread ThreadContext& context) {
  const float top = simd_max(value);
  const float total = simd_sum(value == -INFINITY ? 0.0f : exp(value - top));
  if (context.simd_lane_id == 0) {
    reduce[2 * context.simdgroup_index] = top;
    reduce[2 * context.simdgroup_index + 1] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float overall = -INFINITY;
  for (uint group = 0; group < HEAD_SIMDGROUPS; ++group) {
    overall = max(overall, reduce[2 * group]);
  }
  float sum = 0.0f;
  for (uint group = 0; group < HEAD_SIMDGROUPS; ++group) {
    const float part_top = reduce[2 * group];
    sum += part_top == -INFINITY ? 0.0f : reduce[2 * group + 1] * exp(part_top - overall);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return overall + log(sum);
}

template <bool MAXIMUM>
METAL_FUNC float head_combine(float left, float right) {
  return MAXIMUM ? max(left, right) : left + right;
}

template <bool MAXIMUM>
METAL_FUNC float head_reduce16(thread const float* values, uint lane) {
  const bool high16 = (lane & 16u) != 0u;
  const bool high8 = (lane & 8u) != 0u;
  const bool high4 = (lane & 4u) != 0u;
  const bool high2 = (lane & 2u) != 0u;
  float eight[8];
  METAL_PRAGMA_UNROLL
  for (uint index = 0; index < 8u; ++index) {
    eight[index] = head_combine<MAXIMUM>(
        high16 ? values[8u + index] : values[index],
        simd_shuffle_xor(high16 ? values[index] : values[8u + index], ushort(16))
    );
  }
  float four[4];
  METAL_PRAGMA_UNROLL
  for (uint index = 0; index < 4u; ++index) {
    four[index] = head_combine<MAXIMUM>(
        high8 ? eight[4u + index] : eight[index],
        simd_shuffle_xor(high8 ? eight[index] : eight[4u + index], ushort(8))
    );
  }
  float two[2];
  METAL_PRAGMA_UNROLL
  for (uint index = 0; index < 2u; ++index) {
    two[index] = head_combine<MAXIMUM>(
        high4 ? four[2u + index] : four[index],
        simd_shuffle_xor(high4 ? four[index] : four[2u + index], ushort(4))
    );
  }
  const float one =
      head_combine<MAXIMUM>(high2 ? two[1] : two[0], simd_shuffle_xor(high2 ? two[0] : two[1], ushort(2)));
  return head_combine<MAXIMUM>(one, simd_shuffle_xor(one, ushort(1)));
}

METAL_FUNC void head_dense(
    const device bfloat* weights,
    const device bfloat* biases,
    const threadgroup float* input,
    uint input_stride,
    threadgroup float* output,
    uint output_stride,
    uint inputs,
    uint outputs,
    uint rows,
    bool activate,
    bool accumulate,
    const thread ThreadContext& context
) {
  static_assert(HEAD_DENSE_UNITS * CHILDREN_MAX == 16u, "the butterfly reduces sixteen sums");
  const uint lane = context.simd_lane_id;
  for (uint first = HEAD_DENSE_UNITS * context.simdgroup_index; first < outputs;
       first += HEAD_DENSE_UNITS * HEAD_SIMDGROUPS) {
    float sums[HEAD_DENSE_UNITS * CHILDREN_MAX];
    METAL_PRAGMA_UNROLL
    for (uint index = 0; index < HEAD_DENSE_UNITS * CHILDREN_MAX; ++index) {
      sums[index] = 0.0f;
    }
    for (uint column = 4u * lane; column < inputs; column += 4u * METAL_SIMD_SIZE) {
      float4 values[CHILDREN_MAX];
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < CHILDREN_MAX; ++row) {
        values[row] =
            row < rows
                ? float4(*reinterpret_cast<const threadgroup packed_float4*>(input + row * input_stride + column))
                : float4(0.0f);
      }
      METAL_PRAGMA_UNROLL
      for (uint unit = 0; unit < HEAD_DENSE_UNITS; ++unit) {
        const float4 weight =
            first + unit < outputs
                ? float4(*reinterpret_cast<const device bfloat4*>(weights + (first + unit) * inputs + column))
                : float4(0.0f);
        METAL_PRAGMA_UNROLL
        for (uint row = 0; row < CHILDREN_MAX; ++row) {
          sums[unit * CHILDREN_MAX + row] += dot(weight, values[row]);
        }
      }
    }
    const float total = head_reduce16<false>(sums, lane);
    const uint index = (lane >> 1u) & 15u;
    const uint unit = first + index / CHILDREN_MAX;
    const uint row = index % CHILDREN_MAX;
    if ((lane & 1u) == 0u && unit < outputs && row < rows) {
      float value = total + float(biases[unit]);
      value = activate ? activate_silu(value) : value;
      threadgroup float* target = output + row * output_stride + unit;
      *target = accumulate ? *target + value : value;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

METAL_FUNC void head_layer_norm(
    const threadgroup float* input,
    threadgroup float* output,
    uint stride,
    uint count,
    uint rows,
    const device bfloat* scales,
    const device bfloat* biases,
    float epsilon,
    const thread ThreadContext& context
) {
  const uint row = context.simdgroup_index;
  const uint lane = context.simd_lane_id;
  if (row < rows) {
    const threadgroup float* values = input + row * stride;
    float total = 0.0f;
    for (uint index = lane; index < count; index += 32u) {
      total += values[index];
    }
    const float mean = simd_sum(total) / float(count);
    float spread = 0.0f;
    for (uint index = lane; index < count; index += 32u) {
      spread += (values[index] - mean) * (values[index] - mean);
    }
    const float scale = rsqrt(simd_sum(spread) / float(count) + epsilon);
    for (uint index = lane; index < count; index += 32u) {
      output[row * stride + index] = (values[index] - mean) * scale * float(scales[index]) + float(biases[index]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

METAL_FUNC float head_simd_dot(const device bfloat* weights, const threadgroup float* values, uint count, uint lane) {
  float total = 0.0f;
  for (uint index = lane; index < count; index += 32u) {
    total += float(weights[index]) * values[index];
  }
  return simd_sum(total);
}
