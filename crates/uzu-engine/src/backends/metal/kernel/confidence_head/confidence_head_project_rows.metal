#include <metal_stdlib>
#include "../common/dsl.h"
#include "confidence_head.h"

using namespace metal;

#define CH_PROJECT_UNITS 8u

// dense(layer_norm(row)) of every row in f32: the drafter rows of a step (outputs in f32) and, at load, the projected
// token table (outputs in bf16). One threadgroup per (row, block of CH_PROJECT_UNITS outputs), so that a step's few
// rows still spread over the whole GPU: each recomputes its row's statistics, then its threads split the row and
// carry the block's sums. The CPU twin is the reference.
template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(ConfidenceHeadProjectRows)(
    const device T* rows,
    const device float* norm_scales,
    const device float* norm_biases,
    const device float* weights,
    const device float* biases,
    device float* output OPTIONAL(!output_bf16),
    device bfloat* output_half OPTIONAL(output_bf16),
    constant uint& row_count,
    constant float& epsilon,
    const bool output_bf16 SPECIALIZE,
    const uint width SPECIALIZE,
    const uint outputs SPECIALIZE,
    threadgroup float shared[CH_SIMDGROUPS],
    threadgroup float partial[CH_SIMDGROUPS * CH_PROJECT_UNITS],
    const ThreadContext thread_context,
    const uint group GROUPS(row_count * outputs.div_ceil(CH_PROJECT_UNITS)),
    const uint lid THREADS(CH_THREADS)
) {
  const uint blocks = (outputs + CH_PROJECT_UNITS - 1u) / CH_PROJECT_UNITS;
  const uint row = group / blocks, first = (group % blocks) * CH_PROJECT_UNITS;
  const device T* values = rows + row * width;
  float local = 0.0f;
  for (uint index = lid; index < width; index += CH_THREADS) {
    local += float(values[index]);
  }
  const float mean = confidence_head_sum(local, shared, thread_context) / float(width);
  float squares = 0.0f;
  for (uint index = lid; index < width; index += CH_THREADS) {
    const float centred = float(values[index]) - mean;
    squares += centred * centred;
  }
  const float inverse = 1.0f / sqrt(confidence_head_sum(squares, shared, thread_context) / float(width) + epsilon);

  float sums[CH_PROJECT_UNITS];
  METAL_PRAGMA_UNROLL
  for (uint unit = 0; unit < CH_PROJECT_UNITS; ++unit) {
    sums[unit] = 0.0f;
  }
  for (uint index = lid; index < width; index += CH_THREADS) {
    const float normed = (float(values[index]) - mean) * inverse * norm_scales[index] + norm_biases[index];
    METAL_PRAGMA_UNROLL
    for (uint unit = 0; unit < CH_PROJECT_UNITS; ++unit) {
      if (first + unit < outputs) {
        sums[unit] = fma(weights[(first + unit) * width + index], normed, sums[unit]);
      }
    }
  }
  METAL_PRAGMA_UNROLL
  for (uint unit = 0; unit < CH_PROJECT_UNITS; ++unit) {
    const float group_sum = simd_sum(sums[unit]);
    if (thread_context.simd_lane_id == 0) {
      partial[thread_context.simdgroup_index * CH_PROJECT_UNITS + unit] = group_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < CH_PROJECT_UNITS && first + lid < outputs) {
    float value = biases[first + lid];
    for (uint group_index = 0; group_index < CH_SIMDGROUPS; ++group_index) {
      value += partial[group_index * CH_PROJECT_UNITS + lid];
    }
    if (output_bf16) {
      output_half[row * outputs + first + lid] = bfloat(value);
    } else {
      output[row * outputs + first + lid] = value;
    }
  }
}
