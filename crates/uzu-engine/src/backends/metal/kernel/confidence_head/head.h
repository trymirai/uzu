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

template <uint SIMDGROUPS = HEAD_SIMDGROUPS>
METAL_FUNC float head_log_sum_exp(float value, threadgroup float* reduce, const thread ThreadContext& context) {
  static_assert(SIMDGROUPS <= 32u, "a lane per simdgroup");
  const float top = simd_max(value);
  const float total = simd_sum(value == -INFINITY ? 0.0f : exp(value - top));
  if (context.simd_lane_id == 0) {
    reduce[2 * context.simdgroup_index] = top;
    reduce[2 * context.simdgroup_index + 1] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool owned = context.simd_lane_id < SIMDGROUPS;
  const float part_top = owned ? reduce[2 * context.simd_lane_id] : -INFINITY;
  const float part_total = owned ? reduce[2 * context.simd_lane_id + 1] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float overall = simd_max(part_top);
  return overall + log(simd_sum(part_top == -INFINITY ? 0.0f : part_total * exp(part_top - overall)));
}

template <bool MAXIMUM>
METAL_FUNC float head_combine(float left, float right) {
  return MAXIMUM ? max(left, right) : left + right;
}

template <bool MAXIMUM, uint COUNT, ushort LANES = 32>
METAL_FUNC float head_reduce(thread float* values, uint lane) {
  static_assert(COUNT <= LANES && LANES <= 32u, "a lane group per value");
  if constexpr (COUNT > 1u) {
    constexpr ushort mask = LANES / 2u;
    const bool high = (lane & mask) != 0u;
    METAL_PRAGMA_UNROLL
    for (uint index = 0; index < COUNT / 2u; ++index) {
      const float kept = high ? values[COUNT / 2u + index] : values[index];
      const float sent = high ? values[index] : values[COUNT / 2u + index];
      values[index] = head_combine<MAXIMUM>(kept, simd_shuffle_xor(sent, mask));
    }
    return head_reduce<MAXIMUM, COUNT / 2u, mask>(values, lane);
  } else {
    float value = values[0];
    METAL_PRAGMA_UNROLL
    for (ushort mask = LANES / 2u; mask > 0u; mask /= 2u) {
      value = head_combine<MAXIMUM>(value, simd_shuffle_xor(value, mask));
    }
    return value;
  }
}

template <ushort LANES>
METAL_FUNC void head_dense_lanes(
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
  constexpr uint parts = 32u / LANES;
  const uint lane = context.simd_lane_id % LANES;
  for (uint first = HEAD_DENSE_UNITS * (parts * context.simdgroup_index + context.simd_lane_id / LANES);
       first < outputs;
       first += HEAD_DENSE_UNITS * parts * HEAD_SIMDGROUPS) {
    float sums[HEAD_DENSE_UNITS * CHILDREN_MAX];
    METAL_PRAGMA_UNROLL
    for (uint index = 0; index < HEAD_DENSE_UNITS * CHILDREN_MAX; ++index) {
      sums[index] = 0.0f;
    }
    for (uint column = 4u * lane; column < inputs; column += 4u * LANES) {
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
    const float total = head_reduce<false, HEAD_DENSE_UNITS * CHILDREN_MAX, LANES>(sums, context.simd_lane_id);
    const uint index = LANES == 32 ? (lane >> 1u) & 15u : lane;
    const uint unit = first + index / CHILDREN_MAX;
    const uint row = index % CHILDREN_MAX;
    if ((LANES != 32 || (lane & 1u) == 0u) && unit < outputs && row < rows) {
      float value = total + float(biases[unit]);
      value = activate ? activate_silu(value) : value;
      threadgroup float* target = output + row * output_stride + unit;
      *target = accumulate ? *target + value : value;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
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
  if (inputs <= 64u) {
    head_dense_lanes<16>(
        weights,
        biases,
        input,
        input_stride,
        output,
        output_stride,
        inputs,
        outputs,
        rows,
        activate,
        accumulate,
        context
    );
  } else {
    head_dense_lanes<32>(
        weights,
        biases,
        input,
        input_stride,
        output,
        output_stride,
        inputs,
        outputs,
        rows,
        activate,
        accumulate,
        context
    );
  }
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
