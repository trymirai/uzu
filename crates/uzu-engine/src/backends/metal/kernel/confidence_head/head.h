#pragma once

#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/thread_context.h"
#include "../generated/confidence_head.h"

using namespace metal;
using namespace uzu::confidence_head;

#define HEAD_THREADS 512u
#define HEAD_SIMDGROUPS (HEAD_THREADS / 32u)
#define HEAD_CHILDREN_MAX 4u
#define HEAD_MODEL_MAX 64u
#define HEAD_POOL_HIDDEN_MAX 16u
#define HEAD_EDGE_HIDDEN_MAX 128u
#define HEAD_SLOT_NUMBERS 4u
#define HEAD_EDGE_INPUT_MAX (3u * HEAD_MODEL_MAX + HEAD_SLOT_NUMBERS)
#define HEAD_UNITS_PER_GROUP (HEAD_EDGE_HIDDEN_MAX / HEAD_SIMDGROUPS)
#define HEAD_STEPS_MAX ((HEAD_EDGE_INPUT_MAX + 31u) / 32u)
#define HEAD_NONE 0xffffffffu

METAL_FUNC float2 head_log_sum_exp(float2 value, threadgroup float* reduce, const thread ThreadContext& context) {
  const float2 top = simd_max(value);
  const float2 total = simd_sum(select(exp(value - top), float2(0.0f), value == -INFINITY));
  if (context.simd_lane_id == 0) {
    reduce[4 * context.simdgroup_index] = top.x;
    reduce[4 * context.simdgroup_index + 1] = top.y;
    reduce[4 * context.simdgroup_index + 2] = total.x;
    reduce[4 * context.simdgroup_index + 3] = total.y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float2 overall = -INFINITY;
  for (uint group = 0; group < HEAD_SIMDGROUPS; ++group) {
    overall = max(overall, float2(reduce[4 * group], reduce[4 * group + 1]));
  }
  float2 sum = 0.0f;
  for (uint group = 0; group < HEAD_SIMDGROUPS; ++group) {
    const float2 part_top = float2(reduce[4 * group], reduce[4 * group + 1]);
    const float2 part_total = float2(reduce[4 * group + 2], reduce[4 * group + 3]);
    sum += select(part_total * exp(part_top - overall), float2(0.0f), part_top == -INFINITY);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return overall + log(sum);
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
  const uint group = context.simdgroup_index;
  const uint lane = context.simd_lane_id;
  float4 sums[HEAD_UNITS_PER_GROUP];
  METAL_PRAGMA_UNROLL
  for (uint part = 0; part < HEAD_UNITS_PER_GROUP; ++part) {
    sums[part] = 0.0f;
  }
  METAL_PRAGMA_UNROLL
  for (uint step = 0; step < HEAD_STEPS_MAX; ++step) {
    const uint inner = lane + 32u * step;
    if (inner < inputs) {
      const float4 values = float4(
          input[inner],
          input[input_stride + inner],
          input[2u * input_stride + inner],
          input[3u * input_stride + inner]
      );
      METAL_PRAGMA_UNROLL
      for (uint part = 0; part < HEAD_UNITS_PER_GROUP; ++part) {
        const uint unit = group + HEAD_SIMDGROUPS * part;
        if (unit < outputs) {
          sums[part] += float(weights[unit * inputs + inner]) * values;
        }
      }
    }
  }
  METAL_PRAGMA_UNROLL
  for (uint part = 0; part < HEAD_UNITS_PER_GROUP; ++part) {
    sums[part] = simd_sum(sums[part]);
  }
  const uint part = lane / HEAD_CHILDREN_MAX;
  const uint row = lane % HEAD_CHILDREN_MAX;
  const uint unit = group + HEAD_SIMDGROUPS * part;
  if (part < HEAD_UNITS_PER_GROUP && unit < outputs && row < rows) {
    float4 picked = sums[0];
    METAL_PRAGMA_UNROLL
    for (uint other = 1; other < HEAD_UNITS_PER_GROUP; ++other) {
      picked = other == part ? sums[other] : picked;
    }
    float value = (row == 0 ? picked.x : row == 1 ? picked.y : row == 2 ? picked.z : picked.w) + float(biases[unit]);
    value = activate ? value / (1.0f + exp(-value)) : value;
    threadgroup float* target = output + row * output_stride + unit;
    *target = accumulate ? *target + value : value;
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
