#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../hadamard_transform/hadamard_transform.h"
#include "common/fragment.h"
#include "common/mxu_fragment/ops.h"
#include "common/threadgroup_tile.h"

using namespace metal;

template <ushort VALUES_PER_THREAD>
static inline void qtip_activation_transform_grouped_x128(
    device const bfloat* input,
    device int8_t* output,
    device float* scales,
    device const int32_t* rht_factors,
    uint element_count,
    threadgroup float* partial_max,
    uint group_index,
    uint batch_index,
    uint thread_index,
    const ThreadContext thread_context) {
  constexpr uint GROUP_SIZE = uint(VALUES_PER_THREAD) * 128u;
  const uint group_base = group_index * GROUP_SIZE;
  const uint row_base = batch_index * element_count;
  float values[VALUES_PER_THREAD];
  float local_maximum = 0.0f;
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_THREAD; ++value_index) {
    const uint element_in_row = group_base + uint(value_index) * 128u + thread_index;
    float value = float(input[row_base + element_in_row]);
    value = simdgroup_input_random_hadamard_transform(
        thread_context.simd_lane_id, value, rht_factors[element_in_row]);
    values[value_index] = value;
    local_maximum = max(local_maximum, abs(value));
  }

  const float simdgroup_maximum = simd_max(local_maximum);
  if (thread_context.simd_lane_id == 0) {
    partial_max[thread_context.simdgroup_index] = simdgroup_maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float maximum = partial_max[0];
  METAL_PRAGMA_UNROLL
  for (ushort index = 1; index < 4; ++index) {
    maximum = max(maximum, partial_max[index]);
  }
  const float scale = isfinite(maximum) && maximum > 0.0f ? maximum / 127.0f : 1.0f;
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_THREAD; ++value_index) {
    const uint element_in_row = group_base + uint(value_index) * 128u + thread_index;
    output[row_base + element_in_row] = int8_t(clamp(round(values[value_index] / scale), -127.0f, 127.0f));
  }
  if (thread_index == 0) {
    scales[batch_index * (element_count / GROUP_SIZE) + group_index] = scale;
  }
}

template <ushort SIMD_GROUPS>
static inline void qtip_activation_transform_row(
    device const bfloat* input,
    device int8_t* output,
    device float* scales,
    device const int32_t* rht_factors,
    uint element_count,
    threadgroup float* partial_max,
    uint batch_index,
    uint thread_index,
    const ThreadContext thread_context) {
  const uint row_base = batch_index * element_count;
  const uint block_count = element_count / 32u;
  float local_maximum = 0.0f;
  for (uint block = thread_context.simdgroup_index; block < block_count; block += SIMD_GROUPS) {
    const uint element_in_row = block * 32u + thread_context.simd_lane_id;
    float value = float(input[row_base + element_in_row]);
    value = simdgroup_input_random_hadamard_transform(
        thread_context.simd_lane_id, value, rht_factors[element_in_row]);
    local_maximum = max(local_maximum, abs(value));
  }

  const float simdgroup_maximum = simd_max(local_maximum);
  if (thread_context.simd_lane_id == 0) {
    partial_max[thread_context.simdgroup_index] = simdgroup_maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float maximum = thread_context.simd_lane_id < SIMD_GROUPS
      ? partial_max[thread_context.simd_lane_id]
      : 0.0f;
  maximum = simd_max(maximum);
  const float scale = isfinite(maximum) && maximum > 0.0f ? maximum / 127.0f : 1.0f;
  for (uint block = thread_context.simdgroup_index; block < block_count; block += SIMD_GROUPS) {
    const uint element_in_row = block * 32u + thread_context.simd_lane_id;
    float value = float(input[row_base + element_in_row]);
    value = simdgroup_input_random_hadamard_transform(
        thread_context.simd_lane_id, value, rht_factors[element_in_row]);
    output[row_base + element_in_row] = int8_t(clamp(round(value / scale), -127.0f, 127.0f));
  }
  if (thread_index == 0) {
    scales[batch_index] = scale;
  }
}


template <ushort VALUES_PER_THREAD>
static inline void qtip_activation_transform_row_cached_x512(
    device const bfloat* input,
    device int8_t* output,
    device float* scales,
    device const int32_t* rht_factors,
    uint element_count,
    threadgroup float* partial_max,
    uint batch_index,
    uint thread_index,
    const ThreadContext thread_context) {
  const uint row_base = batch_index * element_count;
  float values[VALUES_PER_THREAD];
  float local_maximum = 0.0f;
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_THREAD; ++value_index) {
    const uint block = thread_context.simdgroup_index + uint(value_index) * 16u;
    const uint element_in_row = block * 32u + thread_context.simd_lane_id;
    float value = float(input[row_base + element_in_row]);
    value = simdgroup_input_random_hadamard_transform(
        thread_context.simd_lane_id, value, rht_factors[element_in_row]);
    values[value_index] = value;
    local_maximum = max(local_maximum, abs(value));
  }

  const float simdgroup_maximum = simd_max(local_maximum);
  if (thread_context.simd_lane_id == 0) {
    partial_max[thread_context.simdgroup_index] = simdgroup_maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float maximum = thread_context.simd_lane_id < 16
      ? partial_max[thread_context.simd_lane_id]
      : 0.0f;
  maximum = simd_max(maximum);
  const float scale = isfinite(maximum) && maximum > 0.0f ? maximum / 127.0f : 1.0f;
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_THREAD; ++value_index) {
    const uint block = thread_context.simdgroup_index + uint(value_index) * 16u;
    const uint element_in_row = block * 32u + thread_context.simd_lane_id;
    output[row_base + element_in_row] =
        int8_t(clamp(round(values[value_index] / scale), -127.0f, 127.0f));
  }
  if (thread_index == 0) {
    scales[batch_index] = scale;
  }
}

#define QTIP_ACTIVATION_TRANSFORM_ROW_CACHED_X512(NAME, ELEMENT_COUNT, VALUES_PER_THREAD) \
KERNEL(NAME)( \
    device const bfloat* input, \
    device int8_t* output, \
    device float* scales, \
    device const int32_t* rht_factors, \
    const constant uint& batch_size, \
    const constant uint& element_count, \
    threadgroup float partial_max[16], \
    const uint batch_index GROUPS(batch_size), \
    const uint thread_index THREADS(512), \
    const ThreadContext thread_context) { \
  (void)ELEMENT_COUNT; \
  qtip_activation_transform_row_cached_x512<VALUES_PER_THREAD>( \
      input, output, scales, rht_factors, element_count, partial_max, \
      batch_index, thread_index, thread_context); \
}

QTIP_ACTIVATION_TRANSFORM_ROW_CACHED_X512(QtipActivationTransformRowCached5120x512, 5120, 10)
QTIP_ACTIVATION_TRANSFORM_ROW_CACHED_X512(QtipActivationTransformRowCached6144x512, 6144, 12)
QTIP_ACTIVATION_TRANSFORM_ROW_CACHED_X512(QtipActivationTransformRowCached17408x512, 17408, 34)

#undef QTIP_ACTIVATION_TRANSFORM_ROW_CACHED_X512

template <ushort VALUES_PER_THREAD>
static inline void qtip_activation_transform_row_cached_padded_x512(
    device const bfloat* input,
    device int8_t* output,
    device float* scales,
    device const int32_t* rht_factors,
    uint active_batch_size,
    uint element_count,
    threadgroup float* partial_max,
    uint batch_index,
    uint thread_index,
    ThreadContext thread_context) {
  if (batch_index >= active_batch_size) {
    for (uint element = thread_index; element < element_count; element += 512u) {
      output[batch_index * element_count + element] = int8_t(0);
    }
    if (thread_index == 0) {
      scales[batch_index] = 1.0f;
    }
    return;
  }
  qtip_activation_transform_row_cached_x512<VALUES_PER_THREAD>(
      input, output, scales, rht_factors, element_count, partial_max,
      batch_index, thread_index, thread_context);
}

#define QTIP_ACTIVATION_TRANSFORM_ROW_CACHED_PADDED_X512(NAME, ELEMENT_COUNT, VALUES_PER_THREAD) \
KERNEL(NAME)( \
    device const bfloat* input, \
    device int8_t* output, \
    device float* scales, \
    device const int32_t* rht_factors, \
    const constant uint& active_batch_size, \
    const constant uint& padded_batch_size, \
    const constant uint& element_count, \
    threadgroup float partial_max[16], \
    const uint batch_index GROUPS(padded_batch_size), \
    const uint thread_index THREADS(512), \
    const ThreadContext thread_context) { \
  (void)ELEMENT_COUNT; \
  qtip_activation_transform_row_cached_padded_x512<VALUES_PER_THREAD>( \
      input, output, scales, rht_factors, active_batch_size, element_count, \
      partial_max, batch_index, thread_index, thread_context); \
}

QTIP_ACTIVATION_TRANSFORM_ROW_CACHED_PADDED_X512(QtipActivationTransformRowCachedPadded5120x512, 5120, 10)
QTIP_ACTIVATION_TRANSFORM_ROW_CACHED_PADDED_X512(QtipActivationTransformRowCachedPadded6144x512, 6144, 12)
QTIP_ACTIVATION_TRANSFORM_ROW_CACHED_PADDED_X512(QtipActivationTransformRowCachedPadded17408x512, 17408, 34)

#undef QTIP_ACTIVATION_TRANSFORM_ROW_CACHED_PADDED_X512

#define QTIP_TRANSITION_BITS 6
#define QTIP_TABLE_ENTRIES 512
#define QTIP_SIMDGROUPS 2

static inline ushort qtip_state(uint high, uint low, uint shift) {
  const ulong window = (ulong(high) << 32u) | ulong(low);
  return ushort((window >> (48u - shift)) & 0xFFFFu);
}

static inline float2 qtip_pair(
    ushort state16,
    threadgroup const half2* table,
    float row_scale) {
  const uint state = uint(state16);
  const uint hashed = state * (state + 31u) & 0xFFFFu;
  const uint index = hashed >> 6u & 511u;
  const float sign = (hashed >> 15u) & 1u ? -1.0f : 1.0f;
  const float2 pair = float2(table[index]);
  return float2(pair.x * sign, pair.y) * row_scale;
}

#define QTIP_ACCUM(I, HIGH, LOW, SHIFT) \
  { \
    const ushort state = qtip_state(HIGH, LOW, SHIFT); \
    const float2 pair = qtip_pair(state, local_table, row_scale); \
    sum += dot(pair, float2(activations[activation_base + group_base + I])); \
  }

template <uint LANES_PER_OUTPUT>
static inline void qtip_cached(
    device const uint* streams,
    device const half2* table,
    device const half2* activations,
    device const float* base_scales,
    device const float* gains,
    device half* output,
    const constant uint& rows,
    const constant uint& groups,
    const constant uint& batch,
    const constant uint& words_per_row,
    threadgroup half2* local_table,
    uint threadgroup_index,
    uint thread_index
) {
  for (uint index = thread_index; index < QTIP_TABLE_ENTRIES; index += QTIP_SIMDGROUPS * 32) {
    local_table[index] = table[index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint simdgroup = thread_index >> 5u;
  const uint lane = thread_index & 31u;
  const uint subgroup = lane / LANES_PER_OUTPUT;
  const uint local_lane = lane % LANES_PER_OUTPUT;
  const uint outputs_per_simdgroup = 32u / LANES_PER_OUTPUT;
  const uint work = (threadgroup_index * QTIP_SIMDGROUPS + simdgroup) * outputs_per_simdgroup + subgroup;
  const bool active = work < rows * batch;
  const uint row = active ? work / batch : 0u;
  const uint token = active ? work - row * batch : 0u;
  const uint lane_groups = groups / LANES_PER_OUTPUT;
  const uint lane_group_base = local_lane * lane_groups;
  const uint activation_base = token * groups;
  const float row_scale = active ? base_scales[row] * gains[row] : 0.0f;

  float sum = 0.0f;
  if (active) {
    for (uint offset = 0u; offset < lane_groups; offset += 16u) {
      const uint group_base = lane_group_base + offset;
      const uint word_base = row * words_per_row + (group_base * QTIP_TRANSITION_BITS >> 5u);
      const uint w0 = streams[word_base];
      const uint w1 = streams[word_base + 1u];
      const uint w2 = streams[word_base + 2u];
      const uint w3 = streams[word_base + 3u];
      QTIP_ACCUM(0u, w0, w1, 0u);
      QTIP_ACCUM(1u, w0, w1, 6u);
      QTIP_ACCUM(2u, w0, w1, 12u);
      QTIP_ACCUM(3u, w0, w1, 18u);
      QTIP_ACCUM(4u, w0, w1, 24u);
      QTIP_ACCUM(5u, w0, w1, 30u);
      QTIP_ACCUM(6u, w1, w2, 4u);
      QTIP_ACCUM(7u, w1, w2, 10u);
      QTIP_ACCUM(8u, w1, w2, 16u);
      QTIP_ACCUM(9u, w1, w2, 22u);
      QTIP_ACCUM(10u, w1, w2, 28u);
      QTIP_ACCUM(11u, w2, w3, 2u);
      QTIP_ACCUM(12u, w2, w3, 8u);
      QTIP_ACCUM(13u, w2, w3, 14u);
      QTIP_ACCUM(14u, w2, w3, 20u);
      QTIP_ACCUM(15u, w2, w3, 26u);
    }
  }

  if (LANES_PER_OUTPUT == 4u && local_lane < 2u) sum += simd_shuffle_down(sum, 2u);
  if (LANES_PER_OUTPUT >= 2u && local_lane == 0u) sum += simd_shuffle_down(sum, 1u);
  if (active && local_lane == 0u) output[work] = half(sum);
}

#define QTIP_ACCUM_BATCH(I, HIGH, LOW, SHIFT) \
  { \
    const ushort state = qtip_state(HIGH, LOW, SHIFT); \
    const float2 pair = qtip_pair(state, local_table, row_scale); \
    const uint group = group_base + I; \
    for (uint token = 0; token < TOKENS; ++token) { \
      sums[token] += dot(pair, float2(activations[token * groups + group])); \
    } \
  }

template <uint TOKENS>
static inline void qtip_cached_batch(
    device const uint* streams,
    device const half2* table,
    device const half2* activations,
    device const float* base_scales,
    device const float* gains,
    device half* output,
    const constant uint& rows,
    const constant uint& groups,
    const constant uint& words_per_row,
    threadgroup half2* local_table,
    uint threadgroup_index,
    uint thread_index
) {
  for (uint index = thread_index; index < QTIP_TABLE_ENTRIES; index += QTIP_SIMDGROUPS * 32) {
    local_table[index] = table[index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint row = threadgroup_index * (QTIP_SIMDGROUPS * 32) + thread_index;
  const bool active = row < rows;
  const float row_scale = active ? base_scales[row] * gains[row] : 0.0f;
  float sums[TOKENS] = {};

  if (active) {
    for (uint group_base = 0; group_base < groups; group_base += 16) {
      const uint word_base = row * words_per_row + (group_base * QTIP_TRANSITION_BITS >> 5u);
      const uint w0 = streams[word_base];
      const uint w1 = streams[word_base + 1u];
      const uint w2 = streams[word_base + 2u];
      const uint w3 = streams[word_base + 3u];
      QTIP_ACCUM_BATCH(0u, w0, w1, 0u);
      QTIP_ACCUM_BATCH(1u, w0, w1, 6u);
      QTIP_ACCUM_BATCH(2u, w0, w1, 12u);
      QTIP_ACCUM_BATCH(3u, w0, w1, 18u);
      QTIP_ACCUM_BATCH(4u, w0, w1, 24u);
      QTIP_ACCUM_BATCH(5u, w0, w1, 30u);
      QTIP_ACCUM_BATCH(6u, w1, w2, 4u);
      QTIP_ACCUM_BATCH(7u, w1, w2, 10u);
      QTIP_ACCUM_BATCH(8u, w1, w2, 16u);
      QTIP_ACCUM_BATCH(9u, w1, w2, 22u);
      QTIP_ACCUM_BATCH(10u, w1, w2, 28u);
      QTIP_ACCUM_BATCH(11u, w2, w3, 2u);
      QTIP_ACCUM_BATCH(12u, w2, w3, 8u);
      QTIP_ACCUM_BATCH(13u, w2, w3, 14u);
      QTIP_ACCUM_BATCH(14u, w2, w3, 20u);
      QTIP_ACCUM_BATCH(15u, w2, w3, 26u);
    }
    for (uint token = 0; token < TOKENS; ++token) {
      output[row * TOKENS + token] = half(sums[token]);
    }
  }
}

static inline uint qtip_murmur_round(uint value) {
  value ^= value >> 16u;
  value *= 0x85ebca6bu;
  return value ^ (value >> 13u);
}

static inline float qtip_1mad(uint value) {
  value = value * 34038481u + 76625530u;
  const uint byte_sum =
      (value & 255u) +
      ((value >> 8u) & 255u) +
      ((value >> 16u) & 255u) +
      (value >> 24u);
  return (float(byte_sum) - 510.0f) * (1.0f / 147.800537109375f);
}

static inline float2 qtip_computed_pair(ushort state16, float row_scale) {
  const uint mixed = qtip_murmur_round(uint(state16));
  const float first = qtip_1mad(mixed);
  const float second = qtip_1mad(mixed ^ 0x9e3779b9u);
  return float2(first, second) * row_scale;
}

#define QTIP_COMPUTED_SIMDGROUPS 2

#define QTIP_COMPUTED_ACCUM_BATCH(I, HIGH, LOW, SHIFT) \
  { \
    const ushort state = qtip_state(HIGH, LOW, SHIFT); \
    const float2 pair = qtip_computed_pair(state, row_scale); \
    const uint group = group_base + I; \
    for (uint token = 0; token < TOKENS; ++token) { \
      sums[token] += dot(pair, float2(activations[token * groups + group])); \
    } \
  }

template <uint TOKENS>
static inline void qtip_computed_batch(
    device const uint* streams,
    device const half2* activations,
    device const float* base_scales,
    device const float* gains,
    device half* output,
    const constant uint& rows,
    const constant uint& groups,
    const constant uint& words_per_row,
    uint threadgroup_index,
    uint thread_index
) {
  const uint row = threadgroup_index * (QTIP_COMPUTED_SIMDGROUPS * 32) + thread_index;
  const bool active = row < rows;
  const float row_scale = active ? base_scales[row] * gains[row] : 0.0f;
  float sums[TOKENS] = {};

  if (active) {
    for (uint group_base = 0; group_base < groups; group_base += 16) {
      const uint word_base = row * words_per_row + (group_base * QTIP_TRANSITION_BITS >> 5u);
      const uint w0 = streams[word_base];
      const uint w1 = streams[word_base + 1u];
      const uint w2 = streams[word_base + 2u];
      const uint w3 = streams[word_base + 3u];
      QTIP_COMPUTED_ACCUM_BATCH(0u, w0, w1, 0u);
      QTIP_COMPUTED_ACCUM_BATCH(1u, w0, w1, 6u);
      QTIP_COMPUTED_ACCUM_BATCH(2u, w0, w1, 12u);
      QTIP_COMPUTED_ACCUM_BATCH(3u, w0, w1, 18u);
      QTIP_COMPUTED_ACCUM_BATCH(4u, w0, w1, 24u);
      QTIP_COMPUTED_ACCUM_BATCH(5u, w0, w1, 30u);
      QTIP_COMPUTED_ACCUM_BATCH(6u, w1, w2, 4u);
      QTIP_COMPUTED_ACCUM_BATCH(7u, w1, w2, 10u);
      QTIP_COMPUTED_ACCUM_BATCH(8u, w1, w2, 16u);
      QTIP_COMPUTED_ACCUM_BATCH(9u, w1, w2, 22u);
      QTIP_COMPUTED_ACCUM_BATCH(10u, w1, w2, 28u);
      QTIP_COMPUTED_ACCUM_BATCH(11u, w2, w3, 2u);
      QTIP_COMPUTED_ACCUM_BATCH(12u, w2, w3, 8u);
      QTIP_COMPUTED_ACCUM_BATCH(13u, w2, w3, 14u);
      QTIP_COMPUTED_ACCUM_BATCH(14u, w2, w3, 20u);
      QTIP_COMPUTED_ACCUM_BATCH(15u, w2, w3, 26u);
    }
    for (uint token = 0; token < TOKENS; ++token) {
      output[row * TOKENS + token] = half(sums[token]);
    }
  }
}

#undef QTIP_COMPUTED_ACCUM_BATCH

static inline uint qtip_state28(uint high, uint low, uint shift) {
  const ulong window = (ulong(high) << 32u) | ulong(low);
  return uint((window >> (36u - shift)) & 0x0fffffffu);
}

static inline float4 qtip_computed_vector4(uint state28, float row_scale) {
  const uint mixed = qtip_murmur_round(state28);
  return float4(
      qtip_1mad(mixed),
      qtip_1mad(mixed ^ 0x9e3779b9u),
      qtip_1mad(mixed ^ 0x3c6ef372u),
      qtip_1mad(mixed ^ 0xdaa66d2bu)) * row_scale;
}

#define QTIP_COMPUTED_V4_ACCUM(I, HIGH, LOW, SHIFT) \
  { \
    const float4 weights = qtip_computed_vector4(qtip_state28(HIGH, LOW, SHIFT), row_scale); \
    const uint activation = 2u * (token * groups + group_base + I); \
    const float4 values = float4(float2(activations[activation]), float2(activations[activation + 1u])); \
    sums[token] += dot(weights, values); \
  }

template <uint TOKENS>
static inline void qtip_computed_v4_batch(
    device const uint* streams,
    device const half2* activations,
    device const float* base_scales,
    device const float* gains,
    device half* output,
    const constant uint& rows,
    const constant uint& groups,
    const constant uint& words_per_row,
    uint threadgroup_index,
    uint thread_index
) {
  const uint row = threadgroup_index * (QTIP_COMPUTED_SIMDGROUPS * 32) + thread_index;
  const bool active = row < rows;
  const float row_scale = active ? base_scales[row] * gains[row] : 0.0f;
  float sums[TOKENS] = {};

  if (active) {
    for (uint group_base = 0; group_base < groups; group_base += 8u) {
      const uint word_base = row * words_per_row + (group_base * 12u >> 5u);
      const uint w0 = streams[word_base];
      const uint w1 = streams[word_base + 1u];
      const uint w2 = streams[word_base + 2u];
      const uint w3 = streams[word_base + 3u];
      for (uint token = 0; token < TOKENS; ++token) {
        QTIP_COMPUTED_V4_ACCUM(0u, w0, w1, 0u);
        QTIP_COMPUTED_V4_ACCUM(1u, w0, w1, 12u);
        QTIP_COMPUTED_V4_ACCUM(2u, w0, w1, 24u);
        QTIP_COMPUTED_V4_ACCUM(3u, w1, w2, 4u);
        QTIP_COMPUTED_V4_ACCUM(4u, w1, w2, 16u);
        QTIP_COMPUTED_V4_ACCUM(5u, w1, w2, 28u);
        QTIP_COMPUTED_V4_ACCUM(6u, w2, w3, 8u);
        QTIP_COMPUTED_V4_ACCUM(7u, w2, w3, 20u);
      }
    }
    for (uint token = 0; token < TOKENS; ++token) {
      output[row * TOKENS + token] = half(sums[token]);
    }
  }
}

#undef QTIP_COMPUTED_V4_ACCUM
#undef QTIP_COMPUTED_SIMDGROUPS

#define QTIP_MMA_ROWS 32
#define QTIP_MMA_DEPTH 32
#define QTIP_MMA_SIMDGROUPS_ROWS 2
#define QTIP_MMA_A_STRIDE 34

template <uint COLS, uint SIMD_COLS, uint THREADS, uint B_STRIDE>
static inline void qtip_tiled(
    device const uint* streams,
    device const half2* table,
    device const half2* activations,
    device const float* base_scales,
    device const float* gains,
    device half* output,
    uint groups,
    uint batch,
    uint words_per_row,
    threadgroup half2* local_table,
    threadgroup half* a_shared,
    threadgroup half* b_shared,
    uint row_tile,
    uint token_tile,
    uint thread_index,
    const thread ThreadContext& thread_context
) {
  using Tile = uzu::matmul::ThreadgroupTile<
      half,
      half,
      half,
      QTIP_MMA_ROWS,
      COLS,
      QTIP_MMA_DEPTH,
      QTIP_MMA_SIMDGROUPS_ROWS,
      SIMD_COLS,
      false,
      false,
      QTIP_MMA_A_STRIDE,
      B_STRIDE,
      float>;

  for (uint index = thread_index; index < QTIP_TABLE_ENTRIES; index += THREADS) {
    local_table[index] = table[index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint row_base = row_tile * QTIP_MMA_ROWS;
  const uint token_base = token_tile * COLS;
  thread Tile accumulator(thread_context);

  for (uint k_base = 0; k_base < groups * 2u; k_base += QTIP_MMA_DEPTH) {
    if (thread_index < QTIP_MMA_ROWS) {
      const uint row = row_base + thread_index;
      const uint group_base = k_base >> 1u;
      const uint word_base = row * words_per_row + (group_base * QTIP_TRANSITION_BITS >> 5u);
      const uint w0 = streams[word_base];
      const uint w1 = streams[word_base + 1u];
      const uint w2 = streams[word_base + 2u];
      const uint w3 = streams[word_base + 3u];
      threadgroup half* weight_row = a_shared + thread_index * QTIP_MMA_A_STRIDE;
#define QTIP_STAGE_PAIR(I, HIGH, LOW, SHIFT) \
  { \
    const float2 pair = qtip_pair(qtip_state(HIGH, LOW, SHIFT), local_table, 1.0f); \
    weight_row[2u * I] = half(pair.x); \
    weight_row[2u * I + 1u] = half(pair.y); \
  }
      QTIP_STAGE_PAIR(0u, w0, w1, 0u);
      QTIP_STAGE_PAIR(1u, w0, w1, 6u);
      QTIP_STAGE_PAIR(2u, w0, w1, 12u);
      QTIP_STAGE_PAIR(3u, w0, w1, 18u);
      QTIP_STAGE_PAIR(4u, w0, w1, 24u);
      QTIP_STAGE_PAIR(5u, w0, w1, 30u);
      QTIP_STAGE_PAIR(6u, w1, w2, 4u);
      QTIP_STAGE_PAIR(7u, w1, w2, 10u);
      QTIP_STAGE_PAIR(8u, w1, w2, 16u);
      QTIP_STAGE_PAIR(9u, w1, w2, 22u);
      QTIP_STAGE_PAIR(10u, w1, w2, 28u);
      QTIP_STAGE_PAIR(11u, w2, w3, 2u);
      QTIP_STAGE_PAIR(12u, w2, w3, 8u);
      QTIP_STAGE_PAIR(13u, w2, w3, 14u);
      QTIP_STAGE_PAIR(14u, w2, w3, 20u);
      QTIP_STAGE_PAIR(15u, w2, w3, 26u);
#undef QTIP_STAGE_PAIR
    }

    const device half* activation_values = reinterpret_cast<const device half*>(activations);
    for (uint index = thread_index; index < QTIP_MMA_DEPTH * COLS; index += THREADS) {
      const uint local_k = index / COLS;
      const uint local_token = index - local_k * COLS;
      b_shared[local_k * B_STRIDE + local_token] =
          activation_values[(token_base + local_token) * groups * 2u + k_base + local_k];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    accumulator.matmul(a_shared, b_shared);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  accumulator.for_each_output([&](ushort row_offset, ushort, ushort, thread float& value) {
    const uint row = row_base + accumulator.simdgroup_row_offset + row_offset;
    value *= base_scales[row] * gains[row];
  });
  accumulator.store_result(output + row_base * batch + token_base, int(batch));
}

#define QTIP_MXU_ROWS 32
#define QTIP_MXU_COLS 32
#define QTIP_MXU_DEPTH 32
#define QTIP_MXU_THREADS 64
#define QTIP_MXU_A_STRIDE 34
#define QTIP_MXU_B_STRIDE 34


#define QTIP_DECODE_SIMDGROUPS 4
#define QTIP_DECODE_THREADS (QTIP_DECODE_SIMDGROUPS * 32)

#undef QTIP_DECODE_THREADS
#undef QTIP_DECODE_SIMDGROUPS

#undef QTIP_ACCUM
#undef QTIP_ACCUM_BATCH
#undef QTIP_SIMDGROUPS
#undef QTIP_TABLE_ENTRIES
#undef QTIP_TRANSITION_BITS

static inline ushort qtip_gaussian_fixture_state(
    device const uchar* row_codes,
    uint group,
    uint transition_bits) {
  const uint bit = group * transition_bits;
  const uint byte = bit >> 3u;
  const uint shift = bit & 7u;
  const uint window =
      (uint(row_codes[byte]) << 16u) |
      (uint(row_codes[byte + 1u]) << 8u) |
      uint(row_codes[byte + 2u]);
  return ushort((window >> (8u - shift)) & 0xFFFFu);
}

static inline ushort2 qtip_gaussian_fixture_state_pair_k2(
    device const uchar* row_codes,
    uint group) {
  const uint bit = group * 4u;
  const uint byte = bit >> 3u;
  const uint shift = bit & 7u;
  const uint window =
      (uint(row_codes[byte]) << 16u) |
      (uint(row_codes[byte + 1u]) << 8u) |
      uint(row_codes[byte + 2u]);
  return ushort2(
      ushort((window >> (8u - shift)) & 0xFFFFu),
      ushort((window >> (4u - shift)) & 0xFFFFu));
}

static inline ushort2 qtip_gaussian_fixture_state_pair_k2_aligned(
    device const uchar* row_codes,
    uint byte) {
  const packed_uchar4 bytes =
      *reinterpret_cast<device const packed_uchar4*>(row_codes + byte);
  const uint byte0 = uint(bytes.x);
  const uint byte1 = uint(bytes.y);
  const uint byte2 = uint(bytes.z);
  return ushort2(
      ushort((byte0 << 8u) | byte1),
      ushort((byte0 << 12u) | (byte1 << 4u) | (byte2 >> 4u)));
}

static inline ushort2 qtip_gaussian_fixture_state_pair_k3(
    device const uchar* row_codes,
    uint group) {
  const uint bit = group * 6u;
  const uint byte = bit >> 3u;
  const uint shift = bit & 7u;
  const uint window =
      (uint(row_codes[byte]) << 24u) |
      (uint(row_codes[byte + 1u]) << 16u) |
      (uint(row_codes[byte + 2u]) << 8u) |
      uint(row_codes[byte + 3u]);
  return ushort2(
      ushort((window >> (16u - shift)) & 0xFFFFu),
      ushort((window >> (10u - shift)) & 0xFFFFu));
}

static inline ushort2 qtip_gaussian_fixture_state_pair_k3_packed(
    device const uchar* row_codes,
    uint group) {
  const uint bit = group * 6u;
  const uint byte = bit >> 3u;
  const uint shift = bit & 7u;
  const packed_uchar4 bytes =
      *reinterpret_cast<device const packed_uchar4*>(row_codes + byte);
  const uint window =
      (uint(bytes.x) << 24u) |
      (uint(bytes.y) << 16u) |
      (uint(bytes.z) << 8u) |
      uint(bytes.w);
  return ushort2(
      ushort((window >> (16u - shift)) & 0xFFFFu),
      ushort((window >> (10u - shift)) & 0xFFFFu));
}

static inline ushort qtip_gaussian_physical_state(
    device const uchar* row_codes,
    uint group,
    uint transition_bits) {
  ushort state = ushort(uint(row_codes[0]) | (uint(row_codes[1]) << 8u));
  if (group == 0u) {
    return state;
  }

  const uint states_in_window = (16u + transition_bits - 1u) / transition_bits;
  const uint first_step = group >= states_in_window ? group - states_in_window + 1u : 1u;
  if (first_step > 1u) {
    state = 0;
  }
  const uint symbol_mask = (1u << transition_bits) - 1u;
  for (uint step = first_step; step <= group; ++step) {
    const uint bit = (step - 1u) * transition_bits;
    const uint byte = 2u + (bit >> 3u);
    const uint shift = bit & 7u;
    uint symbol = uint(row_codes[byte]) >> shift;
    if (shift + transition_bits > 8u) {
      symbol |= uint(row_codes[byte + 1u]) << (8u - shift);
    }
    state = ushort((uint(state) << transition_bits) | (symbol & symbol_mask));
  }
  return state;
}

static inline ushort2 qtip_gaussian_physical_state_pair(
    device const uchar* row_codes,
    uint group,
    uint transition_bits) {
  const ushort first =
      qtip_gaussian_physical_state(row_codes, group, transition_bits);
  const uint bit = group * transition_bits;
  const uint byte = 2u + (bit >> 3u);
  const uint shift = bit & 7u;
  uint symbol = uint(row_codes[byte]) >> shift;
  if (shift + transition_bits > 8u) {
    symbol |= uint(row_codes[byte + 1u]) << (8u - shift);
  }
  const uint symbol_mask = (1u << transition_bits) - 1u;
  const ushort second = ushort(
      (uint(first) << transition_bits) | (symbol & symbol_mask));
  return ushort2(first, second);
}

template <uint TRANSITION_BITS>
static inline ushort qtip_gaussian_physical_state_fast(
    device const uchar* row_codes,
    uint group) {
  static_assert(TRANSITION_BITS == 4u || TRANSITION_BITS == 6u);
  constexpr uint history = TRANSITION_BITS == 4u ? 4u : 3u;
  if (group < history) {
    return qtip_gaussian_physical_state(row_codes, group, TRANSITION_BITS);
  }

  const uint bit = (group - history) * TRANSITION_BITS;
  const uint byte = 2u + (bit >> 3u);
  const uint shift = bit & 7u;
  uint window = uint(row_codes[byte]) |
      (uint(row_codes[byte + 1u]) << 8u) |
      (uint(row_codes[byte + 2u]) << 16u);
  window >>= shift;

  if constexpr (TRANSITION_BITS == 4u) {
    const uint nibbles = window & 0xFFFFu;
    return ushort(
        ((nibbles & 0x000Fu) << 12u) |
        ((nibbles & 0x00F0u) << 4u) |
        ((nibbles & 0x0F00u) >> 4u) |
        ((nibbles & 0xF000u) >> 12u));
  } else {
    return ushort(
        ((window & 0x3Fu) << 12u) |
        (((window >> 6u) & 0x3Fu) << 6u) |
        ((window >> 12u) & 0x3Fu));
  }
}

template <uint TRANSITION_BITS>
static inline ushort2 qtip_gaussian_physical_state_pair_fast(
    device const uchar* row_codes,
    uint group) {
  const ushort first =
      qtip_gaussian_physical_state_fast<TRANSITION_BITS>(row_codes, group);
  const uint bit = group * TRANSITION_BITS;
  const uint byte = 2u + (bit >> 3u);
  const uint shift = bit & 7u;
  uint symbol = uint(row_codes[byte]) >> shift;
  if (shift + TRANSITION_BITS > 8u) {
    symbol |= uint(row_codes[byte + 1u]) << (8u - shift);
  }
  const uint symbol_mask = (1u << TRANSITION_BITS) - 1u;
  return ushort2(
      first,
      ushort((uint(first) << TRANSITION_BITS) | (symbol & symbol_mask)));
}

template <bool K2>
static inline void qtip_gaussian_int8_lut_paired_decode(
    device const uchar* codes,
    device const int8_t* codebook,
    device int8_t* output,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint index) {
  const uint group_pairs = groups / 2u;
  if (index >= rows * group_pairs) {
    return;
  }
  const uint row = index / group_pairs;
  const uint first_group = (index - row * group_pairs) * 2u;
  device const uchar* row_codes = codes + row * bytes_per_row;
  ushort2 states;
  if constexpr (K2) {
    states = qtip_gaussian_fixture_state_pair_k2(row_codes, first_group);
  } else {
    states = qtip_gaussian_fixture_state_pair_k3(row_codes, first_group);
  }
  const uint output_base = index * 4u;
  const uint state0_base = uint(states.x) * 2u;
  const uint state1_base = uint(states.y) * 2u;
  output[output_base] = codebook[state0_base];
  output[output_base + 1u] = codebook[state0_base + 1u];
  output[output_base + 2u] = codebook[state1_base];
  output[output_base + 3u] = codebook[state1_base + 1u];
}


static inline float2 qtip_gaussian_fixture_pair(
    device const float2* codebook,
    ushort state,
    half scale,
    half gain,
    float codebook_scale = 1.0f) {
  const float2 scaled = fma(codebook[state], float(scale) * codebook_scale, 0.0f);
  return scaled * float(gain);
}

static inline float2 qtip_gaussian_fixture_pair(
    device const float2* codebook,
    ushort state,
    half scale,
    ushort gain_bf16,
    float codebook_scale = 1.0f) {
  const float2 scaled = fma(codebook[state], float(scale) * codebook_scale, 0.0f);
  const float gain = as_type<float>(uint(gain_bf16) << 16u);
  return scaled * gain;
}

static inline float2 qtip_gaussian_fixture_pair(
    device const half2* codebook,
    ushort state,
    half scale,
    half gain,
    float codebook_scale = 1.0f) {
  const float2 scaled = fma(float2(codebook[state]), float(scale) * codebook_scale, 0.0f);
  return scaled * float(gain);
}

static inline float2 qtip_gaussian_fixture_pair(
    device const char2* codebook,
    ushort state,
    half scale,
    ushort gain_bf16,
    float codebook_scale) {
  const float2 scaled = fma(float2(codebook[state]), float(scale) * codebook_scale, 0.0f);
  const float gain = as_type<float>(uint(gain_bf16) << 16u);
  return scaled * gain;
}

#define QTIP_GAUSSIAN_FIXTURE_THREADS 128

template <uint TOKENS>
static inline void qtip_gaussian_fixture_batch(
    device const uchar* codes,
    device const float2* codebook,
    device const bfloat2* activations,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    const constant uint& rows,
    const constant uint& groups,
    const constant uint& transition_bits,
    const constant uint& bytes_per_row,
    uint threadgroup_index,
    uint thread_index) {
  const uint row = threadgroup_index * QTIP_GAUSSIAN_FIXTURE_THREADS + thread_index;
  if (row >= rows) {
    return;
  }

  device const uchar* row_codes = codes + row * bytes_per_row;
  const half scale = scales[row];
  const half gain = gains[row];
  float sums[TOKENS] = {};
  for (uint group = 0; group < groups; ++group) {
    const ushort state = qtip_gaussian_fixture_state(row_codes, group, transition_bits);
    const float2 pair = qtip_gaussian_fixture_pair(codebook, state, scale, gain);
    for (uint token = 0; token < TOKENS; ++token) {
      sums[token] += dot(pair, float2(activations[token * groups + group]));
    }
  }

  for (uint token = 0; token < TOKENS; ++token) {
    output[token * rows + row] = bfloat(sums[token]);
  }
}

template <uint TOKENS, uint LANES_PER_ROW, uint SIMD_GROUPS, typename TABLE>
static inline void qtip_gaussian_fixture_ksplit_batch(
    device const uchar* codes,
    device const TABLE* codebook,
    device const bfloat2* activations,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    const constant uint& rows,
    const constant uint& groups,
    const constant uint& transition_bits,
    const constant uint& bytes_per_row,
    uint threadgroup_index,
    uint thread_index) {
  const uint simdgroup = thread_index >> 5u;
  const uint lane = thread_index & 31u;
  const uint rows_per_simdgroup = 32u / LANES_PER_ROW;
  const uint row_in_simdgroup = lane / LANES_PER_ROW;
  const uint local_lane = lane % LANES_PER_ROW;
  const uint row = (threadgroup_index * SIMD_GROUPS + simdgroup) * rows_per_simdgroup + row_in_simdgroup;
  if (row >= rows) {
    return;
  }

  device const uchar* row_codes = codes + row * bytes_per_row;
  const half scale = scales[row];
  const half gain = gains[row];
  float sums[TOKENS] = {};
  for (uint group = local_lane; group < groups; group += LANES_PER_ROW) {
    const ushort state = qtip_gaussian_fixture_state(row_codes, group, transition_bits);
    const float2 pair = qtip_gaussian_fixture_pair(codebook, state, scale, gain);
    for (uint token = 0; token < TOKENS; ++token) {
      sums[token] += dot(pair, float2(activations[token * groups + group]));
    }
  }

  for (uint token = 0; token < TOKENS; ++token) {
    for (uint offset = LANES_PER_ROW / 2u; offset > 0u; offset >>= 1u) {
      const float other = simd_shuffle_down(sums[token], offset);
      if (local_lane < offset) {
        sums[token] += other;
      }
    }
    if (local_lane == 0u) {
      output[token * rows + row] = bfloat(sums[token]);
    }
  }
}


template <uint LANES_PER_ROW, uint SIMD_GROUPS, bool K2>
static inline void qtip_gaussian_fixture_paired_batch8(
    device const uchar* codes,
    device const float2* codebook,
    device const bfloat2* activations,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    const constant uint& rows,
    const constant uint& groups,
    const constant uint& bytes_per_row,
    uint threadgroup_index,
    uint thread_index) {
  const uint simdgroup = thread_index >> 5u;
  const uint lane = thread_index & 31u;
  const uint rows_per_simdgroup = 32u / LANES_PER_ROW;
  const uint row_in_simdgroup = lane / LANES_PER_ROW;
  const uint local_lane = lane % LANES_PER_ROW;
  const uint row = (threadgroup_index * SIMD_GROUPS + simdgroup) * rows_per_simdgroup + row_in_simdgroup;
  if (row >= rows) {
    return;
  }

  device const uchar* row_codes = codes + row * bytes_per_row;
  const half scale = scales[row];
  const half gain = gains[row];
  float sums[8] = {};
  for (uint group = local_lane * 2u; group < groups; group += LANES_PER_ROW * 2u) {
    ushort2 states;
    if constexpr (K2) {
      states = qtip_gaussian_fixture_state_pair_k2(row_codes, group);
    } else {
      states = qtip_gaussian_fixture_state_pair_k3(row_codes, group);
    }
    const float2 pair0 = qtip_gaussian_fixture_pair(codebook, states.x, scale, gain);
    const float2 pair1 = qtip_gaussian_fixture_pair(codebook, states.y, scale, gain);
    METAL_PRAGMA_UNROLL
    for (uint token = 0u; token < 8u; ++token) {
      sums[token] += dot(pair0, float2(activations[token * groups + group]));
      sums[token] += dot(pair1, float2(activations[token * groups + group + 1u]));
    }
  }

  METAL_PRAGMA_UNROLL
  for (uint token = 0u; token < 8u; ++token) {
    for (uint offset = LANES_PER_ROW / 2u; offset > 0u; offset >>= 1u) {
      sums[token] += simd_shuffle_down(sums[token], offset);
    }
    if (local_lane == 0u) {
      output[token * rows + row] = bfloat(sums[token]);
    }
  }
}


template <bool ONE_MULTIPLY>
static inline float2 qtip_gaussian_computed_walsh_pair(
    ushort state,
    half scale,
    half gain) {
  uint value;
  if constexpr (ONE_MULTIPLY) {
    value = uint(state) * 0x020762d1u;
  } else {
    value = uint(state) * 0x85ebca6bu;
    value ^= value >> 13u;
    value = value * 34038481u + 76625530u;
  }
  const float b0 = float(value & 255u);
  const float b1 = float((value >> 8u) & 255u);
  const float b2 = float((value >> 16u) & 255u);
  const float b3 = float(value >> 24u);
  const float2 normalized = float2(
      b0 + b1 + b2 + b3 - 510.0f,
      b0 - b1 + b2 - b3) * (1.0f / 147.800537109375f);
  return fma(normalized, float(scale), 0.0f) * float(gain);
}

template <uint TOKENS, uint LANES_PER_ROW, uint SIMD_GROUPS, bool ONE_MULTIPLY>
static inline void qtip_gaussian_computed_walsh_batch(
    device const uchar* codes,
    device const bfloat2* activations,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    const constant uint& rows,
    const constant uint& groups,
    const constant uint& transition_bits,
    const constant uint& bytes_per_row,
    uint threadgroup_index,
    uint thread_index) {
  const uint simdgroup = thread_index >> 5u;
  const uint lane = thread_index & 31u;
  const uint rows_per_simdgroup = 32u / LANES_PER_ROW;
  const uint row_in_simdgroup = lane / LANES_PER_ROW;
  const uint local_lane = lane % LANES_PER_ROW;
  const uint row =
      (threadgroup_index * SIMD_GROUPS + simdgroup) * rows_per_simdgroup +
      row_in_simdgroup;
  if (row >= rows) {
    return;
  }

  device const uchar* row_codes = codes + row * bytes_per_row;
  const half scale = scales[row];
  const half gain = gains[row];
  float sums[TOKENS] = {};
  for (uint group = local_lane; group < groups; group += LANES_PER_ROW) {
    const ushort state = qtip_gaussian_fixture_state(
        row_codes, group, transition_bits);
    const float2 pair = qtip_gaussian_computed_walsh_pair<ONE_MULTIPLY>(state, scale, gain);
    for (uint token = 0; token < TOKENS; ++token) {
      sums[token] += dot(pair, float2(activations[token * groups + group]));
    }
  }

  for (uint token = 0; token < TOKENS; ++token) {
    for (uint offset = LANES_PER_ROW / 2u; offset > 0u; offset >>= 1u) {
      const float other = simd_shuffle_down(sums[token], offset);
      if (local_lane < offset) {
        sums[token] += other;
      }
    }
    if (local_lane == 0u) {
      output[token * rows + row] = bfloat(sums[token]);
    }
  }
}


#define QTIP_GAUSSIAN_FIXTURE_CHUNK_ACCUM(STATE, GROUP) \
  { \
    const float2 pair = qtip_gaussian_fixture_pair(codebook, STATE, scale, gain); \
    for (uint token = 0; token < 8u; ++token) { \
      sums[token] += dot(pair, float2(activations[token * groups + GROUP])); \
    } \
  }

#undef QTIP_GAUSSIAN_FIXTURE_CHUNK_ACCUM

#define QTIP_GAUSSIAN_FIXTURE_STRIDED_ACCUM(STATE, GROUP) \
  { \
    const float2 pair = qtip_gaussian_fixture_pair(codebook, STATE, scale, gain); \
    for (uint token = 0; token < 8u; ++token) { \
      sums[token] += dot(pair, float2(activations[token * groups + GROUP])); \
    } \
  }

#undef QTIP_GAUSSIAN_FIXTURE_STRIDED_ACCUM


template <uint TOKENS, uint LANES_PER_ROW, uint TRANSITION_BITS, uint BYTES_PER_ROW>
static inline void qtip_gaussian_fixture_packed_subgroup_batch(
    device const uchar* codes,
    device const float2* codebook,
    device const bfloat2* activations,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    const constant uint& rows,
    const constant uint& groups,
    uint threadgroup_index,
    uint thread_index) {
  const uint simdgroup = thread_index >> 5u;
  const uint lane = thread_index & 31u;
  const uint rows_per_simdgroup = 32u / LANES_PER_ROW;
  const uint row_in_simdgroup = lane / LANES_PER_ROW;
  const uint local_lane = lane % LANES_PER_ROW;
  const uint row = (threadgroup_index * 4u + simdgroup) * rows_per_simdgroup + row_in_simdgroup;
  if (row >= rows) {
    return;
  }

  device const uchar* row_codes = codes + row * BYTES_PER_ROW;
  const half scale = scales[row];
  const half gain = gains[row];
  float sums[TOKENS] = {};
  for (uint group_base = 0; group_base < groups; group_base += LANES_PER_ROW) {
    const uint bit = group_base * TRANSITION_BITS;
    const uint byte = bit >> 3u;
    uint window = 0u;
    if (local_lane == 0u) {
      window =
          (uint(row_codes[byte]) << 24u) |
          (uint(row_codes[byte + 1u]) << 16u) |
          (uint(row_codes[byte + 2u]) << 8u) |
          uint(row_codes[byte + 3u]);
    }
    window = simd_shuffle(window, lane - local_lane);
    const uint start = (bit & 7u) + local_lane * TRANSITION_BITS;
    const ushort state = ushort((window >> (16u - start)) & 0xFFFFu);
    const float2 pair = qtip_gaussian_fixture_pair(codebook, state, scale, gain);
    const uint group = group_base + local_lane;
    for (uint token = 0; token < TOKENS; ++token) {
      sums[token] += dot(pair, float2(activations[token * groups + group]));
    }
  }

  for (uint token = 0; token < TOKENS; ++token) {
    for (uint offset = LANES_PER_ROW / 2u; offset > 0u; offset >>= 1u) {
      const float other = simd_shuffle_down(sums[token], offset);
      if (local_lane < offset) {
        sums[token] += other;
      }
    }
    if (local_lane == 0u) {
      output[token * rows + row] = bfloat(sums[token]);
    }
  }
}


static inline float qtip_gain_value(half gain) {
  return float(gain);
}

static inline float qtip_gain_value(ushort gain_bf16) {
  return as_type<float>(uint(gain_bf16) << 16u);
}

template <
    uint COLS,
    uint SIMD_COLS,
    uint THREADS,
    uint B_STRIDE,
    uint DECODE_LANES,
    bool COMPUTED_ONE_MUL,
    bool PAIRED_K2,
    bool PAIRED_K3,
    bool PHYSICAL = false,
    typename Codebook,
    typename Gain>
static inline void qtip_gaussian_fixture_mxu(
    device const uchar* codes,
    device const Codebook* codebook,
    device const bfloat2* activations,
    device const half* scales,
    device const Gain* gains,
    device bfloat* output,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint transition_bits,
    threadgroup bfloat* a_shared,
    threadgroup bfloat* b_shared,
    uint row_tile,
    uint token_tile,
    uint thread_index,
    const thread ThreadContext& thread_context,
    float codebook_scale = 1.0f) {
  using Tile = uzu::matmul::ThreadgroupTile<
      bfloat,
      bfloat,
      bfloat,
      QTIP_MMA_ROWS,
      COLS,
      QTIP_MMA_DEPTH,
      QTIP_MMA_SIMDGROUPS_ROWS,
      SIMD_COLS,
      false,
      false,
      QTIP_MMA_A_STRIDE,
      B_STRIDE,
      float>;

  const uint row_base = row_tile * QTIP_MMA_ROWS;
  const uint token_base = token_tile * COLS;
  thread Tile accumulator(thread_context);

  for (uint k_base = 0; k_base < groups * 2u; k_base += QTIP_MMA_DEPTH) {
    if (thread_index < QTIP_MMA_ROWS * DECODE_LANES) {
      const uint local_row = thread_index / DECODE_LANES;
      const uint decode_lane = thread_index - local_row * DECODE_LANES;
      const uint row = row_base + local_row;
      threadgroup bfloat* weight_row = a_shared + local_row * QTIP_MMA_A_STRIDE;
      if (row < rows) {
        device const uchar* row_codes = codes + row * bytes_per_row;
        const uint group_base = k_base >> 1u;
        const half scale = scales[row];
        const Gain gain = gains[row];
        if constexpr (PAIRED_K2 || PAIRED_K3) {
          METAL_PRAGMA_UNROLL
          for (uint pair_index = decode_lane * 2u; pair_index < QTIP_MMA_DEPTH / 2u; pair_index += DECODE_LANES * 2u) {
            ushort2 states;
            if constexpr (PHYSICAL) {
              states = qtip_gaussian_physical_state_pair(
                  row_codes, group_base + pair_index, transition_bits);
            } else if constexpr (PAIRED_K2) {
              states = qtip_gaussian_fixture_state_pair_k2(row_codes, group_base + pair_index);
            } else {
              states = qtip_gaussian_fixture_state_pair_k3(row_codes, group_base + pair_index);
            }
            const float2 pair0 = qtip_gaussian_fixture_pair(codebook, states.x, scale, gain, codebook_scale);
            const float2 pair1 = qtip_gaussian_fixture_pair(codebook, states.y, scale, gain, codebook_scale);
            weight_row[2u * pair_index] = bfloat(pair0.x);
            weight_row[2u * pair_index + 1u] = bfloat(pair0.y);
            weight_row[2u * pair_index + 2u] = bfloat(pair1.x);
            weight_row[2u * pair_index + 3u] = bfloat(pair1.y);
          }
        } else {
          METAL_PRAGMA_UNROLL
          for (uint pair_index = decode_lane; pair_index < QTIP_MMA_DEPTH / 2u; pair_index += DECODE_LANES) {
            const ushort state = PHYSICAL
                ? qtip_gaussian_physical_state(row_codes, group_base + pair_index, transition_bits)
                : qtip_gaussian_fixture_state(row_codes, group_base + pair_index, transition_bits);
            float2 pair;
            if constexpr (COMPUTED_ONE_MUL) {
              pair = qtip_gaussian_computed_walsh_pair<true>(state, scale, gain);
            } else {
              pair = qtip_gaussian_fixture_pair(codebook, state, scale, gain, codebook_scale);
            }
            weight_row[2u * pair_index] = bfloat(pair.x);
            weight_row[2u * pair_index + 1u] = bfloat(pair.y);
          }
        }
      } else {
        for (uint k = decode_lane; k < QTIP_MMA_DEPTH; k += DECODE_LANES) {
          weight_row[k] = bfloat(0.0f);
        }
      }
    }

    const device bfloat* activation_values = reinterpret_cast<const device bfloat*>(activations);
    for (uint index = thread_index; index < QTIP_MMA_DEPTH * COLS; index += THREADS) {
      const uint local_k = index / COLS;
      const uint local_token = index - local_k * COLS;
      b_shared[local_k * B_STRIDE + local_token] =
          activation_values[(token_base + local_token) * groups * 2u + k_base + local_k];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    accumulator.matmul(a_shared, b_shared);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  accumulator.store_result(output + row_base * COLS + token_base, int(COLS));
}


static inline int8_t qtip_gaussian_one_mul_int8(int value) {
  const int quotient = value >> 2;
  const int remainder = value & 3;
  const int rounded = quotient + int(remainder > 2 || (remainder == 2 && (quotient & 1)));
  return int8_t(clamp(rounded, -127, 127));
}

static inline int8_t qtip_gaussian_round_quarter_unclamped(int value) {
  const int quotient = value >> 2;
  const int remainder = value & 3;
  return int8_t(quotient + int(remainder > 2 || (remainder == 2 && (quotient & 1))));
}

static inline int8_t qtip_gaussian_one_mad_int8(uint source) {
  const uint value = source * 34038481u + 76625530u;
  const int byte_sum =
      int(value & 255u) +
      int((value >> 8u) & 255u) +
      int((value >> 16u) & 255u) +
      int(value >> 24u);
  return qtip_gaussian_one_mul_int8(byte_sum - 510);
}

static inline char2 qtip_gaussian_two_1mad_int8_pair(ushort state) {
  const uint mixed = qtip_murmur_round(uint(state));
  return char2(
      qtip_gaussian_one_mad_int8(mixed),
      qtip_gaussian_one_mad_int8(mixed ^ 0x9e3779b9u));
}

static inline char4 qtip_gaussian_two_1mad_int8_quad(ushort state) {
  const uint mixed = qtip_murmur_round(uint(state));
  return char4(
      qtip_gaussian_one_mad_int8(mixed),
      qtip_gaussian_one_mad_int8(mixed ^ 0x9e3779b9u),
      qtip_gaussian_one_mad_int8(mixed ^ 0x3c6ef372u),
      qtip_gaussian_one_mad_int8(mixed ^ 0xdaa66d2bu));
}

static inline char4 qtip_gaussian_two_mul_walsh_int8_quad(ushort state) {
  const uint mixed = qtip_murmur_round(uint(state));
  const uint value0 = mixed * 34038481u + 76625530u;
  const uint value1 = (mixed ^ 0x9e3779b9u) * 34038481u + 76625530u;
  const int a0 = int(value0 & 255u);
  const int a1 = int((value0 >> 8u) & 255u);
  const int a2 = int((value0 >> 16u) & 255u);
  const int a3 = int(value0 >> 24u);
  const int b0 = int(value1 & 255u);
  const int b1 = int((value1 >> 8u) & 255u);
  const int b2 = int((value1 >> 16u) & 255u);
  const int b3 = int(value1 >> 24u);
  return char4(
      qtip_gaussian_one_mul_int8(a0 + a1 + a2 + a3 - 510),
      qtip_gaussian_one_mul_int8(a0 - a1 + a2 - a3),
      qtip_gaussian_one_mul_int8(b0 + b1 + b2 + b3 - 510),
      qtip_gaussian_one_mul_int8(b0 - b1 + b2 - b3));
}

static inline char4 qtip_gaussian_one_mul_nibble_walsh_int8_quad(ushort state) {
  const uint mixed = qtip_murmur_round(uint(state));
  const uint value = mixed * 34038481u + 76625530u;
  const int n0 = int(value & 15u);
  const int n1 = int((value >> 4u) & 15u);
  const int n2 = int((value >> 8u) & 15u);
  const int n3 = int((value >> 12u) & 15u);
  const int n4 = int((value >> 16u) & 15u);
  const int n5 = int((value >> 20u) & 15u);
  const int n6 = int((value >> 24u) & 15u);
  const int n7 = int(value >> 28u);
  return char4(
      int8_t((n0 + n1 + n2 + n3 - 30) * 4),
      int8_t((n0 - n1 + n2 - n3) * 4),
      int8_t((n4 + n5 + n6 + n7 - 30) * 4),
      int8_t((n4 - n5 + n6 - n7) * 4));
}

static inline char4 qtip_gaussian_byte_q2_dither_int8_quad(
    ushort state,
    uint multiplier,
    uint bias) {
  const uint mixed = qtip_murmur_round(uint(state));
  const uint value = mixed * multiplier + bias;
  const uint pairs =
      (value & 0x03030303u) + ((value >> 2u) & 0x03030303u) +
      ((value >> 4u) & 0x03030303u) + ((value >> 6u) & 0x03030303u);
  const uint dither =
      (((value & 0x0f0f0f0fu) * 3u + 0x01010101u) & 0x0f0f0f0fu) << 1u;
  const uint biased = pairs * 10u + dither + 0x35353535u;
  return as_type<char4>(biased ^ 0x80808080u);
}

static inline char2 qtip_gaussian_one_mul_int8_pair_from_value(uint value) {
  const uint even_bytes = value & 0x00ff00ffu;
  const uint odd_bytes = (value >> 8u) & 0x00ff00ffu;
  const uint pair_sums = even_bytes + odd_bytes;
  const uint pair_differences = even_bytes + 0x01000100u - odd_bytes;
  const int sum = int(pair_sums & 0xffffu) + int(pair_sums >> 16u) - 510;
  const int alternating = int(pair_differences & 0xffffu) + int(pair_differences >> 16u) - 512;
  return char2(
      qtip_gaussian_one_mul_int8(sum),
      qtip_gaussian_one_mul_int8(alternating));
}

template <uint STATE_MIXER = 0>
static inline char2 qtip_gaussian_one_mul_int8_pair(ushort state) {
  uint source = uint(state);
  if constexpr (STATE_MIXER == 1) {
    source = qtip_murmur_round(source);
  } else if constexpr (STATE_MIXER == 2) {
    source ^= source << 11u;
    source ^= source >> 17u;
    source ^= source << 5u;
  } else if constexpr (STATE_MIXER == 3) {
    source ^= source << 11u;
    source ^= source >> 3u;
    source ^= source << 3u;
  }
  return qtip_gaussian_one_mul_int8_pair_from_value(source * 0x020762d1u);
}

constant uint QTIP_K2_NIBBLE_PRODUCTS[16] = {
    0x00000000u, 0x020762d1u, 0x040ec5a2u, 0x06162873u,
    0x081d8b44u, 0x0a24ee15u, 0x0c2c50e6u, 0x0e33b3b7u,
    0x103b1688u, 0x12427959u, 0x1449dc2au, 0x16513efbu,
    0x1858a1ccu, 0x1a60049du, 0x1c67676eu, 0x1e6eca3fu,
};

static inline char4 qtip_gaussian_one_mul_int8_adjacent_quad(ushort2 states) {
  const uint value0 = uint(states.x) * 0x020762d1u;
  const uint value1 = (value0 << 4u) + QTIP_K2_NIBBLE_PRODUCTS[uint(states.y) & 15u];
  const char2 pair0 = qtip_gaussian_one_mul_int8_pair_from_value(value0);
  const char2 pair1 = qtip_gaussian_one_mul_int8_pair_from_value(value1);
  return char4(pair0, pair1);
}

static inline char2 qtip_gaussian_split3_one_mul_int8_pair(ushort state) {
  const uint value = uint(state) * 0x020762d1u;
  const int b0 = int(value & 255u);
  const int b1 = int((value >> 8u) & 255u);
  const int b2 = int((value >> 16u) & 255u);
  const int b3 = int(value >> 24u);
  return char2(
      qtip_gaussian_round_quarter_unclamped(b0 + b1 + b2 - 383),
      qtip_gaussian_round_quarter_unclamped(b1 - b2 + b3 - 128));
}

template <uint VALUE_MAP, uint STATE_MIXER>
static inline char2 qtip_gaussian_computed_int8_pair(ushort state) {
  if constexpr (VALUE_MAP == 1) {
    return qtip_gaussian_split3_one_mul_int8_pair(state);
  }
  return qtip_gaussian_one_mul_int8_pair<STATE_MIXER>(state);
}

template <uint VALUE_MAP>
static inline float qtip_gaussian_computed_int8_scale() {
  if constexpr (VALUE_MAP == 1) {
    return 1.0f / 32.002239227f;
  }
  return 4.0f / 147.800537f;
}

static inline char4 qtip_gaussian_one_mul_int8_quad(ushort state) {
  const uint value = uint(state) * 0x020762d1u;
  const int b0 = int(value & 255u);
  const int b1 = int((value >> 8u) & 255u);
  const int b2 = int((value >> 16u) & 255u);
  const int b3 = int(value >> 24u);
  return char4(
      qtip_gaussian_one_mul_int8(b0 + b1 + b2 + b3 - 510),
      qtip_gaussian_one_mul_int8(b0 - b1 + b2 - b3),
      qtip_gaussian_one_mul_int8(b0 + b1 - b2 - b3),
      qtip_gaussian_one_mul_int8(b0 - b1 - b2 + b3));
}

template <uint COLS>
static inline void qtip_gaussian_one_mul_a8_mxu(
    device const uchar* codes,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint transition_bits,
    threadgroup int8_t* weight_shared,
    threadgroup int8_t* activation_shared,
    uint row_tile,
    uint thread_index,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::Fragment<int8_t, COLS / 16, 2, Ops>;
  using RightTile = uzu::matmul::Fragment<
      int8_t, 2, 2, Ops, uzu::matmul::ReadDirect, true>;
  using Accumulator = uzu::matmul::Fragment<int32_t, COLS / 16, 2, Ops>;

  const uint row_base = row_tile * 32u;
  Accumulator accumulator;
  accumulator.clear();

  for (uint k_base = 0; k_base < groups * 2u; k_base += 32u) {
    if (thread_index < 32u) {
      const uint row = row_base + thread_index;
      threadgroup int8_t* weight_row = weight_shared + thread_index * 32u;
      if (row < rows) {
        device const uchar* row_codes = codes + row * bytes_per_row;
        const uint group_base = k_base >> 1u;
        METAL_PRAGMA_UNROLL
        for (uint pair_index = 0; pair_index < 16u; ++pair_index) {
          const ushort state =
              qtip_gaussian_fixture_state(row_codes, group_base + pair_index, transition_bits);
          const char2 pair = qtip_gaussian_one_mul_int8_pair(state);
          weight_row[2u * pair_index] = pair.x;
          weight_row[2u * pair_index + 1u] = pair.y;
        }
      } else {
        METAL_PRAGMA_UNROLL
        for (uint k = 0; k < 32u; ++k) {
          weight_row[k] = int8_t(0);
        }
      }
    }

    for (uint index = thread_index; index < 32u * COLS; index += 32u) {
      const uint local_k = index / COLS;
      const uint token = index - local_k * COLS;
      activation_shared[index] = activations[token * groups * 2u + k_base + local_k];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    LeftTile left_tile;
    RightTile right_tile;
    volatile int mxu_iteration_fence;
    left_tile.load_from(thread_context.simd_lane_id, uzu::matmul::fragment_source(weight_shared, 32));
    right_tile.load_from(thread_context.simd_lane_id, uzu::matmul::fragment_source(activation_shared, COLS));
    uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
    (void)mxu_iteration_fence;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows) {
      const float weight_scale = float(scales[absolute_row]) * float(gains[absolute_row]) * (4.0f / 147.800537f);
      output[absolute_row * COLS + uint(col)] =
          bfloat(float(value) * weight_scale * activation_scales[uint(col)]);
    }
    return value;
  });
}


template <uint COLS>
static inline void qtip_gaussian_one_mul_a8_mxu_direct(
    device const uchar* codes,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint transition_bits,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 2, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<int8_t, 2, COLS / 16, Ops>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 2, COLS / 16, Ops>;

  const uint row_base = row_tile * 32u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  Accumulator accumulator;
  accumulator.clear();

  for (uint k_base = 0; k_base < groups * 2u; k_base += 32u) {
    LeftTile left_tile;
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < 2; ++fragment_row) {
      METAL_PRAGMA_UNROLL
      for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
        thread auto& fragment_values = left_tile.fragment_at(fragment_row, fragment_col);
        METAL_PRAGMA_UNROLL
        for (ushort local_row = 0; local_row < 2; ++local_row) {
          const uint row = row_base + uint(lane_position.y) + uint(fragment_row) * 16u + uint(local_row) * 8u;
          const ushort element_base = local_row * 4u;
          if (row < rows) {
            device const uchar* row_codes = codes + row * bytes_per_row;
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; local_col += 2) {
              const uint col = uint(lane_position.x) + uint(fragment_col) * 16u + uint(local_col);
              const ushort state = qtip_gaussian_fixture_state(
                  row_codes, (k_base + col) >> 1u, transition_bits);
              const char2 pair = qtip_gaussian_one_mul_int8_pair(state);
              fragment_values[element_base + local_col] = pair.x;
              fragment_values[element_base + local_col + 1u] = pair.y;
            }
          } else {
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; ++local_col) {
              fragment_values[element_base + local_col] = int8_t(0);
            }
          }
        }
      }
    }

    RightTile right_tile;
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < 2; ++fragment_row) {
      METAL_PRAGMA_UNROLL
      for (ushort fragment_col = 0; fragment_col < COLS / 16; ++fragment_col) {
        thread auto& fragment_values = right_tile.fragment_at(fragment_row, fragment_col);
        METAL_PRAGMA_UNROLL
        for (ushort local_row = 0; local_row < 2; ++local_row) {
          const uint k = uint(lane_position.y) + uint(fragment_row) * 16u + uint(local_row) * 8u;
          const ushort element_base = local_row * 4u;
          METAL_PRAGMA_UNROLL
          for (ushort local_col = 0; local_col < 4; ++local_col) {
            const uint token = uint(lane_position.x) + uint(fragment_col) * 16u + uint(local_col);
            fragment_values[element_base + local_col] =
                activations[token * groups * 2u + k_base + k];
          }
        }
      }
    }
    uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows) {
      const float weight_scale = float(scales[absolute_row]) * float(gains[absolute_row]) * (4.0f / 147.800537f);
      output[absolute_row * COLS + uint(col)] =
          bfloat(float(value) * weight_scale * activation_scales[uint(col)]);
    }
    return value;
  });
}


static inline uint qtip_gaussian_load_be_u32(
    device const uchar* row_codes,
    uint byte,
    uint bytes_per_row) {
  uint value = 0u;
  METAL_PRAGMA_UNROLL
  for (uint index = 0; index < 4u; ++index) {
    value <<= 8u;
    if (byte + index < bytes_per_row) {
      value |= uint(row_codes[byte + index]);
    }
  }
  return value;
}

static inline ushort qtip_gaussian_packed_state(
    uint word0,
    uint word1,
    uint word2,
    uint word3,
    uint bit) {
  const uint word_index = bit >> 5u;
  const uint shift = bit & 31u;
  const uint high = word_index == 0u ? word0 : (word_index == 1u ? word1 : word2);
  const uint low = word_index == 0u ? word1 : (word_index == 1u ? word2 : word3);
  if (shift <= 16u) {
    return ushort((high >> (16u - shift)) & 0xFFFFu);
  }
  return ushort(((high << (shift - 16u)) | (low >> (48u - shift))) & 0xFFFFu);
}

template <uint COLS, uint TRANSITION_BITS>
static inline void qtip_gaussian_one_mul_a8_mxu_direct_packed(
    device const uchar* codes,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 2, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<
      int8_t, 2, COLS / 16, Ops, uzu::matmul::ReadTranspose>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 2, COLS / 16, Ops>;

  const uint row_base = row_tile * 32u;
  const ushort lane = thread_context.simd_lane_id;
  const ushort leader_lane = lane & ushort(~9u);
  const short2 lane_position = Ops::get_position(lane);
  Accumulator accumulator;
  accumulator.clear();

  for (uint k_base = 0; k_base < groups * 2u; k_base += 32u) {
    const uint group_base = k_base >> 1u;
    const uint stream_byte = group_base * TRANSITION_BITS >> 3u;
    LeftTile left_tile;
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < 2; ++fragment_row) {
      METAL_PRAGMA_UNROLL
      for (ushort local_row = 0; local_row < 2; ++local_row) {
        const uint row = row_base + uint(lane_position.y) + uint(fragment_row) * 16u + uint(local_row) * 8u;
        uint word0 = 0u;
        uint word1 = 0u;
        uint word2 = 0u;
        uint word3 = 0u;
        if (lane_position.x == 0 && row < rows) {
          device const uchar* row_codes = codes + row * bytes_per_row;
          word0 = qtip_gaussian_load_be_u32(row_codes, stream_byte, bytes_per_row);
          word1 = qtip_gaussian_load_be_u32(row_codes, stream_byte + 4u, bytes_per_row);
          word2 = qtip_gaussian_load_be_u32(row_codes, stream_byte + 8u, bytes_per_row);
          if constexpr (TRANSITION_BITS == 6u) {
            word3 = qtip_gaussian_load_be_u32(row_codes, stream_byte + 12u, bytes_per_row);
          }
        }
        word0 = simd_shuffle(word0, leader_lane);
        word1 = simd_shuffle(word1, leader_lane);
        word2 = simd_shuffle(word2, leader_lane);
        if constexpr (TRANSITION_BITS == 6u) {
          word3 = simd_shuffle(word3, leader_lane);
        }

        METAL_PRAGMA_UNROLL
        for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
          thread auto& fragment_values = left_tile.fragment_at(fragment_row, fragment_col);
          const ushort element_base = local_row * 4u;
          if (row < rows) {
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; local_col += 2) {
              const uint col = uint(lane_position.x) + uint(fragment_col) * 16u + uint(local_col);
              const uint group_offset = col >> 1u;
              const ushort state = qtip_gaussian_packed_state(
                  word0, word1, word2, word3, group_offset * TRANSITION_BITS);
              const char2 pair = qtip_gaussian_one_mul_int8_pair(state);
              fragment_values[element_base + local_col] = pair.x;
              fragment_values[element_base + local_col + 1u] = pair.y;
            }
          } else {
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; ++local_col) {
              fragment_values[element_base + local_col] = int8_t(0);
            }
          }
        }
      }
    }

    RightTile right_tile;
    right_tile.load_from(
        lane,
        uzu::matmul::fragment_source(activations + k_base, int(groups * 2u)));
    uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
  }

  accumulator.map_coords(lane, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows) {
      const float weight_scale = float(scales[absolute_row]) * float(gains[absolute_row]) * (4.0f / 147.800537f);
      output[absolute_row * COLS + uint(col)] =
          bfloat(float(value) * weight_scale * activation_scales[uint(col)]);
    }
    return value;
  });
}


template <uint TRANSITION_BITS>
static inline ushort qtip_gaussian_word_state(
    device const uint* row_words,
    uint group) {
  const uint bit = group * TRANSITION_BITS;
  const uint word_index = bit >> 5u;
  const uint shift = bit & 31u;
  const uint high = row_words[word_index];
  if (shift <= 16u) {
    return ushort((high >> (16u - shift)) & 0xFFFFu);
  }
  return ushort(((high << (shift - 16u)) | (row_words[word_index + 1u] >> (48u - shift))) & 0xFFFFu);
}

template <uint COLS, uint TRANSITION_BITS>
static inline void qtip_gaussian_one_mul_a8_mxu_direct_words(
    device const uint* code_words,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    uint rows,
    uint groups,
    uint words_per_row,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 2, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<
      int8_t, 2, COLS / 16, Ops, uzu::matmul::ReadTranspose>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 2, COLS / 16, Ops>;

  const uint row_base = row_tile * 32u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  Accumulator accumulator;
  accumulator.clear();

  for (uint k_base = 0; k_base < groups * 2u; k_base += 32u) {
    LeftTile left_tile;
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < 2; ++fragment_row) {
      METAL_PRAGMA_UNROLL
      for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
        thread auto& fragment_values = left_tile.fragment_at(fragment_row, fragment_col);
        METAL_PRAGMA_UNROLL
        for (ushort local_row = 0; local_row < 2; ++local_row) {
          const uint row = row_base + uint(lane_position.y) + uint(fragment_row) * 16u + uint(local_row) * 8u;
          const ushort element_base = local_row * 4u;
          if (row < rows) {
            device const uint* row_words = code_words + row * words_per_row;
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; local_col += 2) {
              const uint col = uint(lane_position.x) + uint(fragment_col) * 16u + uint(local_col);
              const ushort state = qtip_gaussian_word_state<TRANSITION_BITS>(
                  row_words, (k_base + col) >> 1u);
              const char2 pair = qtip_gaussian_one_mul_int8_pair(state);
              fragment_values[element_base + local_col] = pair.x;
              fragment_values[element_base + local_col + 1u] = pair.y;
            }
          } else {
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; ++local_col) {
              fragment_values[element_base + local_col] = int8_t(0);
            }
          }
        }
      }
    }

    RightTile right_tile;
    right_tile.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(activations + k_base, int(groups * 2u)));
    uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows) {
      const float weight_scale = float(scales[absolute_row]) * float(gains[absolute_row]) * (4.0f / 147.800537f);
      output[absolute_row * COLS + uint(col)] =
          bfloat(float(value) * weight_scale * activation_scales[uint(col)]);
    }
    return value;
  });
}


template <uint COLS, uint PARTITIONS>
static inline void qtip_gaussian_one_mul_a8_mxu_split_k(
    device const uchar* codes,
    device const int8_t* activations,
    device const half* scales,
    device const half* gains,
    device int32_t* partials,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint transition_bits,
    uint row_tile,
    uint partition,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 2, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<
      int8_t, 2, COLS / 16, Ops, uzu::matmul::ReadTranspose>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 2, COLS / 16, Ops>;

  const uint row_base = row_tile * 32u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint chunk_count = (groups + 15u) / 16u;
  const uint chunks_per_partition = (chunk_count + PARTITIONS - 1u) / PARTITIONS;
  const uint first_chunk = partition * chunks_per_partition;
  const uint final_chunk = min(chunk_count, first_chunk + chunks_per_partition);
  Accumulator accumulator;
  accumulator.clear();

  for (uint chunk = first_chunk; chunk < final_chunk; ++chunk) {
    const uint k_base = chunk * 32u;
    LeftTile left_tile;
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < 2; ++fragment_row) {
      METAL_PRAGMA_UNROLL
      for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
        thread auto& fragment_values = left_tile.fragment_at(fragment_row, fragment_col);
        METAL_PRAGMA_UNROLL
        for (ushort local_row = 0; local_row < 2; ++local_row) {
          const uint row = row_base + uint(lane_position.y) + uint(fragment_row) * 16u + uint(local_row) * 8u;
          const ushort element_base = local_row * 4u;
          if (row < rows) {
            device const uchar* row_codes = codes + row * bytes_per_row;
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; local_col += 2) {
              const uint col = uint(lane_position.x) + uint(fragment_col) * 16u + uint(local_col);
              const ushort state = qtip_gaussian_fixture_state(
                  row_codes, (k_base + col) >> 1u, transition_bits);
              const char2 pair = qtip_gaussian_one_mul_int8_pair(state);
              fragment_values[element_base + local_col] = pair.x;
              fragment_values[element_base + local_col + 1u] = pair.y;
            }
          } else {
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; ++local_col) {
              fragment_values[element_base + local_col] = int8_t(0);
            }
          }
        }
      }
    }

    RightTile right_tile;
    right_tile.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(activations + k_base, int(groups * 2u)));
    uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows) {
      partials[(partition * rows + absolute_row) * COLS + uint(col)] = value;
    }
    return value;
  });
  (void)scales;
  (void)gains;
}

template <uint COLS, uint PARTITIONS>
static inline void qtip_gaussian_one_mul_a8_mxu_reduce(
    device const int32_t* partials,
    device const float* activation_scales,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    uint rows,
    uint index) {
  const uint element_count = rows * COLS;
  if (index >= element_count) {
    return;
  }
  const uint row = index / COLS;
  const uint col = index - row * COLS;
  int32_t sum = 0;
  METAL_PRAGMA_UNROLL
  for (uint partition = 0; partition < PARTITIONS; ++partition) {
    sum += partials[(partition * rows + row) * COLS + col];
  }
  const float weight_scale = float(scales[row]) * float(gains[row]) * (4.0f / 147.800537f);
  output[index] = bfloat(float(sum) * weight_scale * activation_scales[col]);
}

template <
    uint COLS,
    uint OUTPUT_COLS,
    uint PARTITIONS,
    bool PAIRED_K2 = false,
    bool PAIRED_K3 = false,
    uint STATE_MIXER = 0,
    bool TWO_1MAD = false,
    uint ROW_FRAGMENTS = 2,
    uint ROWS_PER_TILE = 32,
    bool PACKED_K3_LOAD = false,
    uint VALUE_MAP = 0,
    bool K2_ADJACENT_MAP = false,
    bool PRECOMBINED_SCALE = false>
static inline void qtip_gaussian_one_mul_a8_mxu_fused_split_k(
    device const uchar* codes,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    threadgroup int32_t* partials,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint transition_bits,
    uint row_tile,
    uint token_base,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, ROW_FRAGMENTS, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<
      int8_t, 2, COLS / 16, Ops, uzu::matmul::ReadTranspose>;
  using Accumulator = uzu::matmul::Fragment<int32_t, ROW_FRAGMENTS, COLS / 16, Ops>;

  const uint partition = thread_context.simdgroup_index;
  const uint row_base = row_tile * ROWS_PER_TILE;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint chunk_count = (groups + 15u) / 16u;
  const uint chunks_per_partition = (chunk_count + PARTITIONS - 1u) / PARTITIONS;
  const uint first_chunk = partition * chunks_per_partition;
  const uint final_chunk = min(chunk_count, first_chunk + chunks_per_partition);
  Accumulator accumulator;
  accumulator.clear();

  for (uint chunk = first_chunk; chunk < final_chunk; ++chunk) {
    const uint k_base = chunk * 32u;
    LeftTile left_tile;
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
      METAL_PRAGMA_UNROLL
      for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
        thread auto& fragment_values = left_tile.fragment_at(fragment_row, fragment_col);
        METAL_PRAGMA_UNROLL
        for (ushort local_row = 0; local_row < 2; ++local_row) {
          const uint row = row_base + uint(lane_position.y) + uint(fragment_row) * 16u + uint(local_row) * 8u;
          const ushort element_base = local_row * 4u;
          if (row < rows) {
            device const uchar* row_codes = codes + row * bytes_per_row;
            if constexpr (PAIRED_K2 || PAIRED_K3) {
              const uint col = uint(lane_position.x) + uint(fragment_col) * 16u;
              ushort2 states;
              if constexpr (PAIRED_K2) {
                states = qtip_gaussian_fixture_state_pair_k2_aligned(
                    row_codes, (k_base + col) >> 2u);
              } else {
                if constexpr (PACKED_K3_LOAD) {
                  states = qtip_gaussian_fixture_state_pair_k3_packed(
                      row_codes, (k_base + col) >> 1u);
                } else {
                  states = qtip_gaussian_fixture_state_pair_k3(
                      row_codes, (k_base + col) >> 1u);
                }
              }
              char2 pair0;
              char2 pair1;
              if constexpr (K2_ADJACENT_MAP) {
                const char4 pairs = qtip_gaussian_one_mul_int8_adjacent_quad(states);
                pair0 = pairs.xy;
                pair1 = pairs.zw;
              } else {
                pair0 = TWO_1MAD
                    ? qtip_gaussian_two_1mad_int8_pair(states.x)
                    : qtip_gaussian_computed_int8_pair<VALUE_MAP, STATE_MIXER>(states.x);
                pair1 = TWO_1MAD
                    ? qtip_gaussian_two_1mad_int8_pair(states.y)
                    : qtip_gaussian_computed_int8_pair<VALUE_MAP, STATE_MIXER>(states.y);
              }
              fragment_values[element_base] = pair0.x;
              fragment_values[element_base + 1u] = pair0.y;
              fragment_values[element_base + 2u] = pair1.x;
              fragment_values[element_base + 3u] = pair1.y;
            } else {
              METAL_PRAGMA_UNROLL
              for (ushort local_col = 0; local_col < 4; local_col += 2) {
                const uint col = uint(lane_position.x) + uint(fragment_col) * 16u + uint(local_col);
                const ushort state = qtip_gaussian_fixture_state(
                    row_codes, (k_base + col) >> 1u, transition_bits);
                const char2 pair = TWO_1MAD
                    ? qtip_gaussian_two_1mad_int8_pair(state)
                    : qtip_gaussian_computed_int8_pair<VALUE_MAP, STATE_MIXER>(state);
                fragment_values[element_base + local_col] = pair.x;
                fragment_values[element_base + local_col + 1u] = pair.y;
              }
            }
          } else {
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; ++local_col) {
              fragment_values[element_base + local_col] = int8_t(0);
            }
          }
        }
      }
    }

    RightTile right_tile;
    right_tile.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(activations + token_base * groups * 2u + k_base, int(groups * 2u)));
    uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
  }

  if (partition != 0u) {
    accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
      partials[((partition - 1u) * ROWS_PER_TILE + uint(row)) * COLS + uint(col)] = value;
      return value;
    });
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (partition == 0u) {
    accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
      int32_t sum = value;
      METAL_PRAGMA_UNROLL
      for (uint other = 1u; other < PARTITIONS; ++other) {
        sum += partials[((other - 1u) * ROWS_PER_TILE + uint(row)) * COLS + uint(col)];
      }
      const uint absolute_row = row_base + uint(row);
      if (absolute_row < rows) {
        const float weight_scale = PRECOMBINED_SCALE
            ? reinterpret_cast<device const float*>(scales)[absolute_row]
            : float(scales[absolute_row]) * float(gains[absolute_row]) *
                qtip_gaussian_computed_int8_scale<VALUE_MAP>();
        output[absolute_row * OUTPUT_COLS + token_base + uint(col)] =
            bfloat(float(sum) * weight_scale * activation_scales[token_base + uint(col)]);
      }
      return sum;
    });
  }
}

template <uint COLS, uint PARTITIONS, bool K2>
static inline void qtip_gaussian_int8_lut_a8_mxu_fused_split_k(
    device const uchar* codes,
    device const int8_t* codebook,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    threadgroup int32_t* partials,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 2, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<
      int8_t, 2, COLS / 16, Ops, uzu::matmul::ReadTranspose>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 2, COLS / 16, Ops>;

  const uint partition = thread_context.simdgroup_index;
  const uint row_base = row_tile * 32u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint chunk_count = (groups + 15u) / 16u;
  const uint chunks_per_partition = (chunk_count + PARTITIONS - 1u) / PARTITIONS;
  const uint first_chunk = partition * chunks_per_partition;
  const uint final_chunk = min(chunk_count, first_chunk + chunks_per_partition);
  Accumulator accumulator;
  accumulator.clear();

  for (uint chunk = first_chunk; chunk < final_chunk; ++chunk) {
    const uint k_base = chunk * 32u;
    LeftTile left_tile;
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < 2; ++fragment_row) {
      METAL_PRAGMA_UNROLL
      for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
        thread auto& fragment_values = left_tile.fragment_at(fragment_row, fragment_col);
        METAL_PRAGMA_UNROLL
        for (ushort local_row = 0; local_row < 2; ++local_row) {
          const uint row = row_base + uint(lane_position.y) + uint(fragment_row) * 16u + uint(local_row) * 8u;
          const ushort element_base = local_row * 4u;
          if (row < rows) {
            device const uchar* row_codes = codes + row * bytes_per_row;
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; local_col += 2) {
              const uint col = uint(lane_position.x) + uint(fragment_col) * 16u + uint(local_col);
              const ushort state = qtip_gaussian_fixture_state(
                  row_codes, (k_base + col) >> 1u, K2 ? 4u : 6u);
              const uint pair_index = uint(state) * 2u;
              fragment_values[element_base + local_col] = codebook[pair_index];
              fragment_values[element_base + local_col + 1u] = codebook[pair_index + 1u];
            }
          } else {
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; ++local_col) {
              fragment_values[element_base + local_col] = int8_t(0);
            }
          }
        }
      }
    }

    RightTile right_tile;
    right_tile.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(activations + k_base, int(groups * 2u)));
    uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
  }

  if (partition != 0u) {
    accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
      partials[((partition - 1u) * 32u + uint(row)) * COLS + uint(col)] = value;
      return value;
    });
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (partition == 0u) {
    accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
      int32_t sum = value;
      METAL_PRAGMA_UNROLL
      for (uint other = 1u; other < PARTITIONS; ++other) {
        sum += partials[((other - 1u) * 32u + uint(row)) * COLS + uint(col)];
      }
      const uint absolute_row = row_base + uint(row);
      if (absolute_row < rows) {
        const float weight_scale = float(scales[absolute_row]) * float(gains[absolute_row]) * codebook_scale;
        output[absolute_row * COLS + uint(col)] =
            bfloat(float(sum) * weight_scale * activation_scales[uint(col)]);
      }
      return sum;
    });
  }
}

template <
    uint COLS,
    bool K2,
    bool COMPUTED = false,
    bool TWO_1MAD = false,
    uint ROW_SIMDGROUPS = 1>
static inline void qtip_gaussian_int8_lut_a8_mxu_staged(
    device const uchar* codes,
    device const int8_t* codebook,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    threadgroup int8_t* weight_shared,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint row_tile,
    uint thread_index,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::Fragment<int8_t, COLS / 16, 2, Ops>;
  using RightTile = uzu::matmul::Fragment<
      int8_t, 2, 2, Ops, uzu::matmul::ReadDirect, true>;
  using Accumulator = uzu::matmul::Fragment<int32_t, COLS / 16, 2, Ops>;

  const uint simdgroup_index = thread_context.simdgroup_index;
  const uint row_base =
      row_tile * (ROW_SIMDGROUPS * 32u) + simdgroup_index * 32u;
  threadgroup int8_t* simdgroup_weights =
      weight_shared + simdgroup_index * 32u * 32u;
  device const char2* codebook_pairs = reinterpret_cast<device const char2*>(codebook);
  Accumulator accumulator;
  accumulator.clear();

  for (uint k_base = 0; k_base < groups * 2u; k_base += 32u) {
    const uint row = row_base + thread_context.simd_lane_id;
    threadgroup int8_t* weight_row =
        simdgroup_weights + thread_context.simd_lane_id * 32u;
    if (row < rows) {
      device const uchar* row_codes = codes + row * bytes_per_row;
      METAL_PRAGMA_UNROLL
      for (uint local_col = 0; local_col < 32u; local_col += 4u) {
        ushort2 states;
        if constexpr (K2) {
          states = qtip_gaussian_fixture_state_pair_k2_aligned(
              row_codes, (k_base + local_col) >> 2u);
        } else {
          states = qtip_gaussian_fixture_state_pair_k3(
              row_codes, (k_base + local_col) >> 1u);
        }
        char2 pair0;
        char2 pair1;
        if constexpr (COMPUTED) {
          pair0 = TWO_1MAD
              ? qtip_gaussian_two_1mad_int8_pair(states.x)
              : qtip_gaussian_one_mul_int8_pair(states.x);
          pair1 = TWO_1MAD
              ? qtip_gaussian_two_1mad_int8_pair(states.y)
              : qtip_gaussian_one_mul_int8_pair(states.y);
        } else {
          pair0 = codebook_pairs[states.x];
          pair1 = codebook_pairs[states.y];
        }
        weight_row[local_col] = pair0.x;
        weight_row[local_col + 1u] = pair0.y;
        weight_row[local_col + 2u] = pair1.x;
        weight_row[local_col + 3u] = pair1.y;
      }
    } else {
      METAL_PRAGMA_UNROLL
      for (uint local_col = 0; local_col < 32u; ++local_col) {
        weight_row[local_col] = int8_t(0);
      }
    }

    if constexpr (ROW_SIMDGROUPS == 1) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    LeftTile left_tile;
    RightTile right_tile;
    volatile int mxu_iteration_fence;
    left_tile.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(
            activations + k_base, int(groups * 2u)));
    right_tile.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(simdgroup_weights, 32));
    uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
    (void)mxu_iteration_fence;
    if constexpr (ROW_SIMDGROUPS == 1) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short token, short row, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows) {
      const float value_scale = COMPUTED ? (4.0f / 147.800537f) : codebook_scale;
      const float weight_scale =
          float(scales[absolute_row]) * float(gains[absolute_row]) * value_scale;
      output[absolute_row * COLS + uint(token)] =
          bfloat(float(value) * weight_scale * activation_scales[uint(token)]);
    }
    return value;
  });
}

template <uint THREAD_COUNT, bool K2>
static inline void qtip_gaussian_computed_v2_a8_mxu_cooperative_b16(
    device const uchar* codes,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    threadgroup int8_t* weight_shared,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint row_tile,
    uint thread_index,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::Fragment<int8_t, 1, 2, Ops>;
  using RightTile = uzu::matmul::Fragment<
      int8_t, 2, 2, Ops, uzu::matmul::ReadDirect, true>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 1, 2, Ops>;

  const uint row_base = row_tile * 32u;
  Accumulator accumulator;
  accumulator.clear();

  for (uint k_base = 0; k_base < groups * 2u; k_base += 32u) {
    for (uint item = thread_index; item < 256u; item += THREAD_COUNT) {
      const uint local_row = item >> 3u;
      const uint pair_group = item & 7u;
      const uint row = row_base + local_row;
      threadgroup int8_t* destination = weight_shared + local_row * 32u + pair_group * 4u;
      if (row < rows) {
        device const uchar* row_codes = codes + row * bytes_per_row;
        ushort2 states;
        if constexpr (K2) {
          states = qtip_gaussian_fixture_state_pair_k2_aligned(
              row_codes, (k_base >> 2u) + pair_group);
        } else {
          states = qtip_gaussian_fixture_state_pair_k3(
              row_codes, (k_base >> 1u) + pair_group * 2u);
        }
        const char2 pair0 = qtip_gaussian_two_1mad_int8_pair(states.x);
        const char2 pair1 = qtip_gaussian_two_1mad_int8_pair(states.y);
        destination[0] = pair0.x;
        destination[1] = pair0.y;
        destination[2] = pair1.x;
        destination[3] = pair1.y;
      } else {
        *reinterpret_cast<threadgroup int32_t*>(destination) = 0;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_context.simdgroup_index == 0u) {
      LeftTile left_tile;
      RightTile right_tile;
      left_tile.load_from(
          thread_context.simd_lane_id,
          uzu::matmul::fragment_source(
              activations + k_base, int(groups * 2u)));
      right_tile.load_from(
          thread_context.simd_lane_id,
          uzu::matmul::fragment_source(weight_shared, 32));
      uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (thread_context.simdgroup_index == 0u) {
    accumulator.map_coords(thread_context.simd_lane_id, [&](short token, short row, int32_t value) {
      const uint absolute_row = row_base + uint(row);
      if (absolute_row < rows) {
        const float weight_scale =
            float(scales[absolute_row]) * float(gains[absolute_row]) * (4.0f / 147.800537f);
        output[absolute_row * 16u + uint(token)] =
            bfloat(float(value) * weight_scale * activation_scales[uint(token)]);
      }
      return value;
    });
  }
}

template <
    uint COLS,
    uint ROW_SIMDGROUPS,
    uint ROW_FRAGMENTS,
    bool K2,
    bool COMPUTED,
    bool TWO_1MAD = false,
    bool PRECOMBINED_SCALE = false,
    bool PHYSICAL = false,
    typename Gain = half,
    bool BATCH_ROWS = false>
static inline void qtip_gaussian_int8_lut_a8_mxu_direct(
    device const uchar* codes,
    device const int8_t* codebook,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const Gain* gains,
    device bfloat* output,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, ROW_FRAGMENTS, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<
      int8_t, 2, COLS / 16, Ops, uzu::matmul::ReadTranspose>;
  using Accumulator = uzu::matmul::Fragment<int32_t, ROW_FRAGMENTS, COLS / 16, Ops>;

  const uint row_base =
      row_tile * (ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u) +
      thread_context.simdgroup_index * ROW_FRAGMENTS * 16u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint chunk_count = (groups + 15u) / 16u;
  device const char2* codebook_pairs =
      reinterpret_cast<device const char2*>(codebook);
  Accumulator accumulator;

  for (uint chunk = 0; chunk < chunk_count; ++chunk) {
    const uint k_base = chunk * 32u;
    LeftTile left_tile;
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
      METAL_PRAGMA_UNROLL
      for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
        thread auto& fragment_values = left_tile.fragment_at(fragment_row, fragment_col);
        METAL_PRAGMA_UNROLL
        for (ushort local_row = 0; local_row < 2; ++local_row) {
          const uint row = row_base + uint(lane_position.y) +
              uint(fragment_row) * 16u + uint(local_row) * 8u;
          const ushort element_base = local_row * 4u;
          if (row < rows) {
            device const uchar* row_codes = codes + row * bytes_per_row;
            const uint col = uint(lane_position.x) + uint(fragment_col) * 16u;
            ushort2 states;
            if constexpr (PHYSICAL) {
              if constexpr (K2) {
                states = qtip_gaussian_physical_state_pair_fast<4u>(
                    row_codes, (k_base + col) >> 1u);
              } else {
                states = qtip_gaussian_physical_state_pair_fast<6u>(
                    row_codes, (k_base + col) >> 1u);
              }
            } else if constexpr (K2) {
              states = qtip_gaussian_fixture_state_pair_k2_aligned(
                  row_codes, (k_base + col) >> 2u);
            } else {
              states = qtip_gaussian_fixture_state_pair_k3(
                  row_codes, (k_base + col) >> 1u);
            }
            char2 pair0;
            char2 pair1;
            if constexpr (COMPUTED) {
              pair0 = TWO_1MAD
                  ? qtip_gaussian_two_1mad_int8_pair(states.x)
                  : qtip_gaussian_one_mul_int8_pair(states.x);
              pair1 = TWO_1MAD
                  ? qtip_gaussian_two_1mad_int8_pair(states.y)
                  : qtip_gaussian_one_mul_int8_pair(states.y);
            } else {
              pair0 = codebook_pairs[states.x];
              pair1 = codebook_pairs[states.y];
            }
            fragment_values[element_base] = pair0.x;
            fragment_values[element_base + 1u] = pair0.y;
            fragment_values[element_base + 2u] = pair1.x;
            fragment_values[element_base + 3u] = pair1.y;
          } else {
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; ++local_col) {
              fragment_values[element_base + local_col] = int8_t(0);
            }
          }
        }
      }
    }

    RightTile right_tile;
    right_tile.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(activations + k_base, int(groups * 2u)));
    if (chunk == 0u) {
      uzu::matmul::fragment_mm(accumulator, left_tile, right_tile);
    } else {
      uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
    }
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows) {
      const float gain = qtip_gain_value(gains[absolute_row]);
      const float weight_scale = PRECOMBINED_SCALE
          ? reinterpret_cast<device const float*>(scales)[absolute_row]
          : float(scales[absolute_row]) * gain *
              (COMPUTED ? (4.0f / 147.800537f) : codebook_scale);
      const uint output_index = BATCH_ROWS
          ? uint(col) * rows + absolute_row
          : absolute_row * COLS + uint(col);
      output[output_index] =
          bfloat(float(value) * weight_scale * activation_scales[uint(col)]);
    }
    return value;
  });
}

static inline ushort qtip_gaussian_fixture_state_v4_restarted(
    device const uchar* row_codes,
    uint group) {
  const uint block = group >> 4u;
  const uint group_in_block = group & 15u;
  const uint byte = block * 17u + group_in_block;
  return ushort((uint(row_codes[byte]) << 8u) | uint(row_codes[byte + 1u]));
}

static inline ushort qtip_gaussian_physical_state_v4_restarted(
    device const uchar* row_codes,
    uint group) {
  const uint block = group >> 4u;
  const uint group_in_block = group & 15u;
  device const uchar* sequence = row_codes + block * 17u;
  if (group_in_block == 0u) {
    return ushort(uint(sequence[0]) | (uint(sequence[1]) << 8u));
  }
  if (group_in_block == 1u) {
    return ushort((uint(sequence[0]) << 8u) | uint(sequence[2]));
  }
  return ushort(
      (uint(sequence[group_in_block]) << 8u) |
      uint(sequence[group_in_block + 1u]));
}

static inline ushort qtip_gaussian_state_v4_connected(
    device const uchar* row_codes,
    uint group) {
  return ushort((uint(row_codes[group]) << 8u) | uint(row_codes[group + 1u]));
}

template <
    uint COLS,
    uint ROW_SIMDGROUPS,
    uint ROW_FRAGMENTS,
    bool COMPUTED,
    bool TWO_1MAD = false,
    bool TWO_MUL_WALSH = false,
    bool NIBBLE_WALSH = false,
    bool BYTE_Q2_DITHER = false,
    bool CONNECTED = false,
    bool PHYSICAL = false,
    bool BATCH_ROWS = false>
static inline void qtip_gaussian_int8_lut_a8_mxu_direct_v4_restarted(
    device const uchar* codes,
    device const int8_t* codebook,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, ROW_FRAGMENTS, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<
      int8_t, 2, COLS / 16, Ops, uzu::matmul::ReadTranspose>;
  using Accumulator = uzu::matmul::Fragment<int32_t, ROW_FRAGMENTS, COLS / 16, Ops>;

  const uint row_base =
      row_tile * (ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u) +
      thread_context.simdgroup_index * ROW_FRAGMENTS * 16u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint chunk_count = (groups + 7u) / 8u;
  device const char4* codebook_vectors =
      reinterpret_cast<device const char4*>(codebook);
  uint byte_q2_multiplier = 0u;
  uint byte_q2_bias = 0u;
  float byte_q2_scale = 0.0f;
  if constexpr (BYTE_Q2_DITHER) {
    device const uint* parameters = reinterpret_cast<device const uint*>(codebook);
    byte_q2_multiplier = parameters[0];
    byte_q2_bias = parameters[1];
    byte_q2_scale = as_type<float>(parameters[2]);
  }
  Accumulator accumulator;

  for (uint chunk = 0; chunk < chunk_count; ++chunk) {
    const uint k_base = chunk * 32u;
    LeftTile left_tile;
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
      METAL_PRAGMA_UNROLL
      for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
        thread auto& fragment_values = left_tile.fragment_at(fragment_row, fragment_col);
        METAL_PRAGMA_UNROLL
        for (ushort local_row = 0; local_row < 2; ++local_row) {
          const uint row = row_base + uint(lane_position.y) +
              uint(fragment_row) * 16u + uint(local_row) * 8u;
          const ushort element_base = local_row * 4u;
          if (row < rows) {
            device const uchar* row_codes = codes + row * bytes_per_row;
            const uint col = uint(lane_position.x) + uint(fragment_col) * 16u;
            const uint group = (k_base + col) >> 2u;
            ushort state;
            if constexpr (PHYSICAL) {
              state = qtip_gaussian_physical_state_v4_restarted(row_codes, group);
            } else {
              state = CONNECTED
                  ? qtip_gaussian_state_v4_connected(row_codes, group)
                  : qtip_gaussian_fixture_state_v4_restarted(row_codes, group);
            }
            char4 values;
            if constexpr (COMPUTED) {
              if constexpr (TWO_1MAD) {
                values = qtip_gaussian_two_1mad_int8_quad(state);
              } else if constexpr (TWO_MUL_WALSH) {
                values = qtip_gaussian_two_mul_walsh_int8_quad(state);
              } else if constexpr (NIBBLE_WALSH) {
                values = qtip_gaussian_one_mul_nibble_walsh_int8_quad(state);
              } else if constexpr (BYTE_Q2_DITHER) {
                values = qtip_gaussian_byte_q2_dither_int8_quad(
                    state, byte_q2_multiplier, byte_q2_bias);
              } else {
                values = qtip_gaussian_one_mul_int8_quad(state);
              }
            } else {
              values = codebook_vectors[state];
            }
            fragment_values[element_base] = values.x;
            fragment_values[element_base + 1u] = values.y;
            fragment_values[element_base + 2u] = values.z;
            fragment_values[element_base + 3u] = values.w;
          } else {
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; ++local_col) {
              fragment_values[element_base + local_col] = int8_t(0);
            }
          }
        }
      }
    }

    RightTile right_tile;
    right_tile.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(activations + k_base, int(groups * 4u)));
    if (chunk == 0u) {
      uzu::matmul::fragment_mm(accumulator, left_tile, right_tile);
    } else {
      uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
    }
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows) {
      const float gain = as_type<float>(uint(gains_bf16[absolute_row]) << 16u);
      const float value_scale = BYTE_Q2_DITHER
          ? byte_q2_scale
          : (COMPUTED ? (4.0f / 147.800537f) : codebook_scale);
      const float weight_scale =
          float(scales[absolute_row]) * gain * value_scale;
      const uint output_index = BATCH_ROWS
          ? uint(col) * rows + absolute_row
          : absolute_row * COLS + uint(col);
      output[output_index] =
          bfloat(float(value) * weight_scale * activation_scales[uint(col)]);
    }
    return value;
  });
}

template <uint COLS, uint ROW_SIMDGROUPS, bool K2 = false>
static inline void qtip_gaussian_one_mul_a8_mxu_paired_k3_row_split2(
    device const uchar* codes,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 1, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<
      int8_t, 2, COLS / 16, Ops, uzu::matmul::ReadTranspose>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 1, COLS / 16, Ops>;

  const uint row_base = row_tile * (ROW_SIMDGROUPS * 16u) + thread_context.simdgroup_index * 16u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint chunk_count = (groups + 15u) / 16u;
  Accumulator accumulator;
  accumulator.clear();

  for (uint chunk = 0; chunk < chunk_count; ++chunk) {
    const uint k_base = chunk * 32u;
    LeftTile left_tile;
    METAL_PRAGMA_UNROLL
    for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
      thread auto& fragment_values = left_tile.fragment_at(0, fragment_col);
      METAL_PRAGMA_UNROLL
      for (ushort local_row = 0; local_row < 2; ++local_row) {
        const uint row = row_base + uint(lane_position.y) + uint(local_row) * 8u;
        const ushort element_base = local_row * 4u;
        if (row < rows) {
          device const uchar* row_codes = codes + row * bytes_per_row;
          const uint col = uint(lane_position.x) + uint(fragment_col) * 16u;
          ushort2 states;
          if constexpr (K2) {
            states = qtip_gaussian_fixture_state_pair_k2_aligned(
                row_codes, (k_base + col) >> 2u);
          } else {
            states = qtip_gaussian_fixture_state_pair_k3(
                row_codes, (k_base + col) >> 1u);
          }
          const char2 pair0 = qtip_gaussian_one_mul_int8_pair(states.x);
          const char2 pair1 = qtip_gaussian_one_mul_int8_pair(states.y);
          fragment_values[element_base] = pair0.x;
          fragment_values[element_base + 1u] = pair0.y;
          fragment_values[element_base + 2u] = pair1.x;
          fragment_values[element_base + 3u] = pair1.y;
        } else {
          METAL_PRAGMA_UNROLL
          for (ushort local_col = 0; local_col < 4; ++local_col) {
            fragment_values[element_base + local_col] = int8_t(0);
          }
        }
      }
    }

    RightTile right_tile;
    right_tile.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(activations + k_base, int(groups * 2u)));
    uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows) {
      const float weight_scale = float(scales[absolute_row]) * float(gains[absolute_row]) * (4.0f / 147.800537f);
      output[absolute_row * COLS + uint(col)] =
          bfloat(float(value) * weight_scale * activation_scales[uint(col)]);
    }
    return value;
  });
}

template <uint COLS, uint ROW_SIMDGROUPS, uint SCALE_CHUNKS, bool K2 = false>
static inline void qtip_gaussian_one_mul_a8_mxu_paired_k3_row_split_grouped(
    device const uchar* codes,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 1, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<
      int8_t, 2, COLS / 16, Ops, uzu::matmul::ReadTranspose>;
  using GroupAccumulator = uzu::matmul::Fragment<int32_t, 1, COLS / 16, Ops>;
  using Accumulator = uzu::matmul::Fragment<float, 1, COLS / 16, Ops>;

  const uint row_base = row_tile * (ROW_SIMDGROUPS * 16u) + thread_context.simdgroup_index * 16u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint chunk_count = (groups + 15u) / 16u;
  const uint scale_group_count = chunk_count / SCALE_CHUNKS;
  Accumulator accumulator;
  accumulator.clear();

  METAL_PRAGMA_NO_UNROLL
  for (uint scale_group = 0; scale_group < scale_group_count; ++scale_group) {
    GroupAccumulator group_accumulator;
    group_accumulator.clear();
    METAL_PRAGMA_NO_UNROLL
    for (uint scale_chunk = 0; scale_chunk < SCALE_CHUNKS; ++scale_chunk) {
      const uint chunk = scale_group * SCALE_CHUNKS + scale_chunk;
      const uint k_base = chunk * 32u;
      LeftTile left_tile;
      METAL_PRAGMA_UNROLL
      for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
        thread auto& fragment_values = left_tile.fragment_at(0, fragment_col);
        METAL_PRAGMA_UNROLL
        for (ushort local_row = 0; local_row < 2; ++local_row) {
          const uint row = row_base + uint(lane_position.y) + uint(local_row) * 8u;
          const ushort element_base = local_row * 4u;
          if (row < rows) {
            device const uchar* row_codes = codes + row * bytes_per_row;
            const uint col = uint(lane_position.x) + uint(fragment_col) * 16u;
            ushort2 states;
            if constexpr (K2) {
              states = qtip_gaussian_fixture_state_pair_k2(
                  row_codes, (k_base + col) >> 1u);
            } else {
              states = qtip_gaussian_fixture_state_pair_k3(
                  row_codes, (k_base + col) >> 1u);
            }
            const char2 pair0 = qtip_gaussian_one_mul_int8_pair(states.x);
            const char2 pair1 = qtip_gaussian_one_mul_int8_pair(states.y);
            fragment_values[element_base] = pair0.x;
            fragment_values[element_base + 1u] = pair0.y;
            fragment_values[element_base + 2u] = pair1.x;
            fragment_values[element_base + 3u] = pair1.y;
          } else {
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; ++local_col) {
              fragment_values[element_base + local_col] = int8_t(0);
            }
          }
        }
      }

      RightTile right_tile;
      right_tile.load_from(
          thread_context.simd_lane_id,
          uzu::matmul::fragment_source(activations + k_base, int(groups * 2u)));
      uzu::matmul::fragment_mma(group_accumulator, left_tile, right_tile);
    }
    float group_scales[Accumulator::COL_FRAGMENTS * Ops::THREAD_ELEMENT_COLS];
    METAL_PRAGMA_UNROLL
    for (ushort fragment_col = 0; fragment_col < Accumulator::COL_FRAGMENTS; ++fragment_col) {
      METAL_PRAGMA_UNROLL
      for (ushort local_col = 0; local_col < Ops::THREAD_ELEMENT_COLS; ++local_col) {
        const uint col = uint(lane_position.x) + uint(fragment_col) * 16u + uint(local_col);
        group_scales[fragment_col * Ops::THREAD_ELEMENT_COLS + local_col] =
            activation_scales[col * scale_group_count + scale_group];
      }
    }

    thread float* accumulator_values = accumulator.elements();
    thread int32_t* group_values = group_accumulator.elements();
    METAL_PRAGMA_UNROLL
    for (ushort fragment_col = 0; fragment_col < Accumulator::COL_FRAGMENTS; ++fragment_col) {
      const ushort fragment_base = fragment_col * Ops::ELEMENTS_PER_THREAD;
      METAL_PRAGMA_UNROLL
      for (ushort local_row = 0; local_row < Ops::THREAD_ELEMENT_ROWS; ++local_row) {
        METAL_PRAGMA_UNROLL
        for (ushort local_col = 0; local_col < Ops::THREAD_ELEMENT_COLS; ++local_col) {
          const ushort index = fragment_base + local_row * Ops::THREAD_ELEMENT_COLS + local_col;
          accumulator_values[index] +=
              float(group_values[index]) * group_scales[fragment_col * Ops::THREAD_ELEMENT_COLS + local_col];
        }
      }
    }
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, float value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows) {
      const float weight_scale = float(scales[absolute_row]) * float(gains[absolute_row]) * (4.0f / 147.800537f);
      output[absolute_row * COLS + uint(col)] = bfloat(value * weight_scale);
    }
    return value;
  });
}

template <uint COLS, uint PARTITIONS>
static inline void qtip_gaussian_one_mul_a8_mxu_atomic_split_k(
    device const uchar* codes,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    threadgroup _atomic<int32_t>* sums,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint transition_bits,
    uint row_tile,
    uint thread_index,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 2, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<
      int8_t, 2, COLS / 16, Ops, uzu::matmul::ReadTranspose>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 2, COLS / 16, Ops>;

  for (uint index = thread_index; index < 32u * COLS; index += PARTITIONS * 32u) {
    atomic_store_explicit(&sums[index], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint partition = thread_context.simdgroup_index;
  const uint row_base = row_tile * 32u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint chunk_count = (groups + 15u) / 16u;
  const uint chunks_per_partition = (chunk_count + PARTITIONS - 1u) / PARTITIONS;
  const uint first_chunk = partition * chunks_per_partition;
  const uint final_chunk = min(chunk_count, first_chunk + chunks_per_partition);
  Accumulator accumulator;
  accumulator.clear();

  for (uint chunk = first_chunk; chunk < final_chunk; ++chunk) {
    const uint k_base = chunk * 32u;
    LeftTile left_tile;
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < 2; ++fragment_row) {
      METAL_PRAGMA_UNROLL
      for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
        thread auto& fragment_values = left_tile.fragment_at(fragment_row, fragment_col);
        METAL_PRAGMA_UNROLL
        for (ushort local_row = 0; local_row < 2; ++local_row) {
          const uint row = row_base + uint(lane_position.y) + uint(fragment_row) * 16u + uint(local_row) * 8u;
          const ushort element_base = local_row * 4u;
          if (row < rows) {
            device const uchar* row_codes = codes + row * bytes_per_row;
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; local_col += 2) {
              const uint col = uint(lane_position.x) + uint(fragment_col) * 16u + uint(local_col);
              const ushort state = qtip_gaussian_fixture_state(
                  row_codes, (k_base + col) >> 1u, transition_bits);
              const char2 pair = qtip_gaussian_one_mul_int8_pair(state);
              fragment_values[element_base + local_col] = pair.x;
              fragment_values[element_base + local_col + 1u] = pair.y;
            }
          } else {
            METAL_PRAGMA_UNROLL
            for (ushort local_col = 0; local_col < 4; ++local_col) {
              fragment_values[element_base + local_col] = int8_t(0);
            }
          }
        }
      }
    }

    RightTile right_tile;
    right_tile.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(activations + k_base, int(groups * 2u)));
    uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    atomic_fetch_add_explicit(&sums[uint(row) * COLS + uint(col)], value, memory_order_relaxed);
    return value;
  });
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (partition == 0u) {
    accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
      const int32_t sum = atomic_load_explicit(&sums[uint(row) * COLS + uint(col)], memory_order_relaxed);
      const uint absolute_row = row_base + uint(row);
      if (absolute_row < rows) {
        const float weight_scale = float(scales[absolute_row]) * float(gains[absolute_row]) * (4.0f / 147.800537f);
        output[absolute_row * COLS + uint(col)] =
            bfloat(float(sum) * weight_scale * activation_scales[uint(col)]);
      }
      return value;
    });
  }
}

template <uint COLS, uint PARTITIONS>
static inline void qtip_gaussian_dense_a8_mxu_fused_split_k(
    device const int8_t* weights,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const half* gains,
    device bfloat* output,
    threadgroup int32_t* partials,
    uint rows,
    uint columns,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 2, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<
      int8_t, 2, COLS / 16, Ops, uzu::matmul::ReadTranspose>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 2, COLS / 16, Ops>;

  const uint partition = thread_context.simdgroup_index;
  const uint row_base = row_tile * 32u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint chunk_count = columns / 32u;
  const uint chunks_per_partition = (chunk_count + PARTITIONS - 1u) / PARTITIONS;
  const uint first_chunk = partition * chunks_per_partition;
  const uint final_chunk = min(chunk_count, first_chunk + chunks_per_partition);
  Accumulator accumulator;
  accumulator.clear();

  for (uint chunk = first_chunk; chunk < final_chunk; ++chunk) {
    const uint k_base = chunk * 32u;
    LeftTile left_tile;
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < 2; ++fragment_row) {
      METAL_PRAGMA_UNROLL
      for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
        thread auto& fragment_values = left_tile.fragment_at(fragment_row, fragment_col);
        METAL_PRAGMA_UNROLL
        for (ushort local_row = 0; local_row < 2; ++local_row) {
          const uint row = row_base + uint(lane_position.y) + uint(fragment_row) * 16u + uint(local_row) * 8u;
          const ushort element_base = local_row * 4u;
          METAL_PRAGMA_UNROLL
          for (ushort local_col = 0; local_col < 4; ++local_col) {
            const uint col = uint(lane_position.x) + uint(fragment_col) * 16u + uint(local_col);
            fragment_values[element_base + local_col] =
                row < rows ? weights[row * columns + k_base + col] : int8_t(0);
          }
        }
      }
    }

    RightTile right_tile;
    right_tile.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(activations + k_base, int(columns)));
    uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
  }

  if (partition != 0u) {
    accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
      partials[((partition - 1u) * 32u + uint(row)) * COLS + uint(col)] = value;
      return value;
    });
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (partition == 0u) {
    accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
      int32_t sum = value;
      METAL_PRAGMA_UNROLL
      for (uint other = 1u; other < PARTITIONS; ++other) {
        sum += partials[((other - 1u) * 32u + uint(row)) * COLS + uint(col)];
      }
      const uint absolute_row = row_base + uint(row);
      if (absolute_row < rows) {
        const float weight_scale = float(scales[absolute_row]) * float(gains[absolute_row]) * (4.0f / 147.800537f);
        output[absolute_row * COLS + uint(col)] =
            bfloat(float(sum) * weight_scale * activation_scales[uint(col)]);
      }
      return sum;
    });
  }
}


#define QTIP_GAUSSIAN_PHYSICAL_Q8_V2_A8_DIRECT_KERNEL(NAME, COLS, ROW_SIMDGROUPS, ROW_FRAGMENTS, K2, BATCH_ROWS) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const uint row_tile GROUPS(rows.div_ceil(ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS(ROW_SIMDGROUPS * 32u), \
    const ThreadContext thread_context \
) { \
  (void)thread_index; \
  qtip_gaussian_int8_lut_a8_mxu_direct< \
      COLS, ROW_SIMDGROUPS, ROW_FRAGMENTS, K2, false, false, false, false, ushort, BATCH_ROWS>( \
      codes, codebook, activations, activation_scales, scales, gains_bf16, output, \
      codebook_scale, rows, groups, bytes_per_row, row_tile, thread_context); \
}

QTIP_GAUSSIAN_PHYSICAL_Q8_V2_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V2A8DirectK2B16, 16, 4, 2, true, false)
QTIP_GAUSSIAN_PHYSICAL_Q8_V2_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V2A8DirectK2B32, 32, 4, 1, true, false)
QTIP_GAUSSIAN_PHYSICAL_Q8_V2_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V2A8DirectK2B64, 64, 4, 1, true, false)
QTIP_GAUSSIAN_PHYSICAL_Q8_V2_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V2A8DirectK3B16, 16, 4, 2, false, false)
QTIP_GAUSSIAN_PHYSICAL_Q8_V2_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V2A8DirectK3B32, 32, 4, 1, false, false)
QTIP_GAUSSIAN_PHYSICAL_Q8_V2_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V2A8DirectK3B64, 64, 4, 1, false, false)
QTIP_GAUSSIAN_PHYSICAL_Q8_V2_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V2A8DirectK2B32BatchRows, 32, 4, 1, true, true)
QTIP_GAUSSIAN_PHYSICAL_Q8_V2_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V2A8DirectK2B64BatchRows, 64, 4, 1, true, true)
QTIP_GAUSSIAN_PHYSICAL_Q8_V2_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V2A8DirectK3B32BatchRows, 32, 4, 1, false, true)
QTIP_GAUSSIAN_PHYSICAL_Q8_V2_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V2A8DirectK3B64BatchRows, 64, 4, 1, false, true)

#undef QTIP_GAUSSIAN_PHYSICAL_Q8_V2_A8_DIRECT_KERNEL


#define QTIP_GAUSSIAN_PHYSICAL_Q8_V4_A8_DIRECT_KERNEL(NAME, COLS, ROW_SIMDGROUPS, ROW_FRAGMENTS, BATCH_ROWS) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const uint row_tile GROUPS(rows.div_ceil(ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS(ROW_SIMDGROUPS * 32u), \
    const ThreadContext thread_context \
) { \
  (void)thread_index; \
  qtip_gaussian_int8_lut_a8_mxu_direct_v4_restarted< \
      COLS, ROW_SIMDGROUPS, ROW_FRAGMENTS, false, false, false, false, false, false, true, BATCH_ROWS>( \
      codes, codebook, activations, activation_scales, scales, gains_bf16, output, \
      codebook_scale, rows, groups, bytes_per_row, row_tile, thread_context); \
}

QTIP_GAUSSIAN_PHYSICAL_Q8_V4_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V4A8DirectB16, 16, 2, 2, false)
QTIP_GAUSSIAN_PHYSICAL_Q8_V4_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V4A8DirectB32, 32, 4, 1, false)
QTIP_GAUSSIAN_PHYSICAL_Q8_V4_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V4A8DirectB64, 64, 4, 1, false)
QTIP_GAUSSIAN_PHYSICAL_Q8_V4_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V4A8DirectB32BatchRows, 32, 4, 1, true)
QTIP_GAUSSIAN_PHYSICAL_Q8_V4_A8_DIRECT_KERNEL(QtipGaussianPhysicalQ8V4A8DirectB64BatchRows, 64, 4, 1, true)

#undef QTIP_GAUSSIAN_PHYSICAL_Q8_V4_A8_DIRECT_KERNEL

template <
    bool TWO_MUL_WALSH = false,
    bool NIBBLE_WALSH = false,
    uint ROW_SIMDGROUPS = 1,
    bool BYTE_Q2_DITHER = false,
    bool CONNECTED = false>
static inline void qtip_gaussian_computed_v4_a8_mxu_staged_b16(
    device const uchar* codes,
    device const int8_t* codebook,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    threadgroup int8_t* weight_shared,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint row_tile,
    uint thread_index,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::Fragment<int8_t, 1, 2, Ops>;
  using RightTile = uzu::matmul::Fragment<
      int8_t, 2, 2, Ops, uzu::matmul::ReadDirect, true>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 1, 2, Ops>;

  const uint simdgroup_index = thread_context.simdgroup_index;
  const uint row_base =
      row_tile * (ROW_SIMDGROUPS * 32u) + simdgroup_index * 32u;
  threadgroup int8_t* simdgroup_weights =
      weight_shared + simdgroup_index * 32u * 32u;
  Accumulator accumulator;
  accumulator.clear();
  uint byte_q2_multiplier = 0u;
  uint byte_q2_bias = 0u;
  float byte_q2_scale = 0.0f;
  if constexpr (BYTE_Q2_DITHER) {
    device const uint* parameters = reinterpret_cast<device const uint*>(codebook);
    byte_q2_multiplier = parameters[0];
    byte_q2_bias = parameters[1];
    byte_q2_scale = as_type<float>(parameters[2]);
  }

  for (uint k_base = 0; k_base < groups * 4u; k_base += 32u) {
    const uint row = row_base + thread_context.simd_lane_id;
    threadgroup int8_t* weight_row =
        simdgroup_weights + thread_context.simd_lane_id * 32u;
    if (row < rows) {
      device const uchar* row_codes = codes + row * bytes_per_row;
      METAL_PRAGMA_UNROLL
      for (uint local_col = 0; local_col < 32u; local_col += 4u) {
        const uint group = (k_base + local_col) >> 2u;
        const ushort state = CONNECTED
            ? qtip_gaussian_state_v4_connected(row_codes, group)
            : qtip_gaussian_fixture_state_v4_restarted(row_codes, group);
        char4 values;
        if constexpr (TWO_MUL_WALSH) {
          values = qtip_gaussian_two_mul_walsh_int8_quad(state);
        } else if constexpr (NIBBLE_WALSH) {
          values = qtip_gaussian_one_mul_nibble_walsh_int8_quad(state);
        } else if constexpr (BYTE_Q2_DITHER) {
          values = qtip_gaussian_byte_q2_dither_int8_quad(
              state, byte_q2_multiplier, byte_q2_bias);
        } else {
          values = qtip_gaussian_two_1mad_int8_quad(state);
        }
        weight_row[local_col] = values.x;
        weight_row[local_col + 1u] = values.y;
        weight_row[local_col + 2u] = values.z;
        weight_row[local_col + 3u] = values.w;
      }
    } else {
      METAL_PRAGMA_UNROLL
      for (uint local_col = 0; local_col < 32u; ++local_col) {
        weight_row[local_col] = int8_t(0);
      }
    }

    if constexpr (ROW_SIMDGROUPS == 1) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    LeftTile left_tile;
    RightTile right_tile;
    left_tile.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(activations + k_base, int(groups * 4u)));
    right_tile.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(simdgroup_weights, 32));
    uzu::matmul::fragment_mma(accumulator, left_tile, right_tile);
    if constexpr (ROW_SIMDGROUPS == 1) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short token, short row, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows) {
      const float gain = as_type<float>(uint(gains_bf16[absolute_row]) << 16u);
      const float value_scale = BYTE_Q2_DITHER
          ? byte_q2_scale
          : (4.0f / 147.800537f);
      const float weight_scale =
          float(scales[absolute_row]) * gain * value_scale;
      output[absolute_row * 16u + uint(token)] =
          bfloat(float(value) * weight_scale * activation_scales[uint(token)]);
    }
    return value;
  });
}


template <
    uint COLS,
    uint SIMD_COLS,
    uint THREAD_COUNT,
    uint B_STRIDE,
    bool PHYSICAL = false,
    typename Codebook = float4>
static inline void qtip_gaussian_fixture_v4_mxu(
    device const uchar* codes,
    device const Codebook* codebook,
    device const bfloat* activations,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    uint rows,
    uint groups,
    uint bytes_per_row,
    threadgroup bfloat* a_shared,
    threadgroup bfloat* b_shared,
    uint row_tile,
    uint token_tile,
    uint thread_index,
    const thread ThreadContext& thread_context,
    float codebook_scale = 1.0f) {
  using Tile = uzu::matmul::ThreadgroupTile<
      bfloat,
      bfloat,
      bfloat,
      QTIP_MMA_ROWS,
      COLS,
      QTIP_MMA_DEPTH,
      QTIP_MMA_SIMDGROUPS_ROWS,
      SIMD_COLS,
      false,
      false,
      QTIP_MMA_A_STRIDE,
      B_STRIDE,
      float>;

  const uint row_base = row_tile * QTIP_MMA_ROWS;
  const uint token_base = token_tile * COLS;
  thread Tile accumulator(thread_context);

  for (uint k_base = 0; k_base < groups * 4u; k_base += QTIP_MMA_DEPTH) {
    if (thread_index < QTIP_MMA_ROWS * 4u) {
      const uint local_row = thread_index / 4u;
      const uint decode_lane = thread_index - local_row * 4u;
      const uint row = row_base + local_row;
      threadgroup bfloat* weight_row = a_shared + local_row * QTIP_MMA_A_STRIDE;
      if (row < rows) {
        device const uchar* row_codes = codes + row * bytes_per_row;
        const uint group_base = k_base >> 2u;
        const float gain = as_type<float>(uint(gains_bf16[row]) << 16u);
        const float row_scale = float(scales[row]) * gain;
        METAL_PRAGMA_UNROLL
        for (uint vector_index = decode_lane; vector_index < QTIP_MMA_DEPTH / 4u; vector_index += 4u) {
          const ushort state = PHYSICAL
              ? qtip_gaussian_physical_state_v4_restarted(row_codes, group_base + vector_index)
              : qtip_gaussian_fixture_state_v4_restarted(row_codes, group_base + vector_index);
          const float4 values = float4(codebook[state]) * (row_scale * codebook_scale);
          const uint element = vector_index * 4u;
          weight_row[element] = bfloat(values.x);
          weight_row[element + 1u] = bfloat(values.y);
          weight_row[element + 2u] = bfloat(values.z);
          weight_row[element + 3u] = bfloat(values.w);
        }
      } else {
        for (uint k = decode_lane; k < QTIP_MMA_DEPTH; k += 4u) {
          weight_row[k] = bfloat(0.0f);
        }
      }
    }

    for (uint index = thread_index; index < QTIP_MMA_DEPTH * COLS; index += THREAD_COUNT) {
      const uint local_k = index / COLS;
      const uint local_token = index - local_k * COLS;
      b_shared[local_k * B_STRIDE + local_token] =
          activations[(token_base + local_token) * groups * 4u + k_base + local_k];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    accumulator.matmul(a_shared, b_shared);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  accumulator.store_result(output + row_base * COLS + token_base, int(COLS));
}


// Exact physical S-checkpoint support. These kernels keep the stored QTIP
// representation intact: full incoherence is applied to activations, the
// Gaussian table is read as FP32, and surfaces follow the frozen S4 ABI.

KERNEL(QtipFullIncoherenceA8)(
    device const bfloat* input,
    device const float* signs,
    device const float* small_q,
    device int8_t* output,
    device float* scales,
    const constant uint& active_batch,
    const constant uint& padded_batch,
    const constant uint& dimension,
    const constant uint& order,
    const constant uint& power,
    threadgroup float values[2048],
    threadgroup float partial_max[16],
    const uint token GROUPS(padded_batch),
    const uint thread_index THREADS(512),
    const ThreadContext thread_context) {
  if (token >= active_batch) {
    for (uint element = thread_index; element < dimension; element += 512u) {
      output[token * dimension + element] = int8_t(0);
    }
    if (thread_index == 0u) {
      scales[token] = 1.0f;
    }
    return;
  }

  float transformed[34];
  float local_maximum = 0.0f;
  const uint values_per_q = power / 512u;
  const float normalization = rsqrt(float(power));

  for (uint q_out = 0u; q_out < order; ++q_out) {
    for (uint h = thread_index; h < power; h += 512u) {
      const uint base = token * dimension + h * order;
      float value = 0.0f;
      METAL_PRAGMA_UNROLL
      for (uint q = 0u; q < order; ++q) {
        value += float(input[base + q]) * signs[h * order + q] * small_q[q_out * order + q];
      }
      values[h] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1u; stride < power; stride <<= 1u) {
      for (uint pair = thread_index; pair < power / 2u; pair += 512u) {
        const uint low = pair & (stride - 1u);
        const uint base = (pair - low) << 1u;
        const uint lhs = base + low;
        const uint rhs = lhs + stride;
        const float a = values[lhs];
        const float b = values[rhs];
        values[lhs] = a + b;
        values[rhs] = a - b;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint local_value = 0u;
    for (uint h = thread_index; h < power; h += 512u) {
      const float value = float(bfloat(values[h] * normalization));
      transformed[q_out * values_per_q + local_value] = value;
      local_maximum = max(local_maximum, abs(value));
      ++local_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float simdgroup_maximum = simd_max(local_maximum);
  if (thread_context.simd_lane_id == 0u) {
    partial_max[thread_context.simdgroup_index] = simdgroup_maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maximum = thread_context.simd_lane_id < 16u
      ? partial_max[thread_context.simd_lane_id]
      : 0.0f;
  maximum = simd_max(maximum);
  const float scale = isfinite(maximum) && maximum > 0.0f ? maximum / 127.0f : 1.0f;

  for (uint q_out = 0u; q_out < order; ++q_out) {
    uint local_value = 0u;
    for (uint h = thread_index; h < power; h += 512u) {
      const float value = transformed[q_out * values_per_q + local_value];
      output[token * dimension + h * order + q_out] =
          int8_t(clamp(round(value / scale), -127.0f, 127.0f));
      ++local_value;
    }
  }
  if (thread_index == 0u) {
    scales[token] = scale;
  }
}

KERNEL(QtipRowsBatchToBatchRowsBf16)(
    device const bfloat* input,
    device bfloat* output,
    const constant uint& active_batch,
    const constant uint& padded_batch,
    const constant uint& rows,
    const uint index AXIS(active_batch * rows, 256)) {
  const uint token = index / rows;
  const uint row = index - token * rows;
  output[index] = input[row * padded_batch + token];
}

KERNEL(QtipRowsBatchToBatchRowsF32)(
    device const bfloat* input,
    device float* output,
    const constant uint& active_batch,
    const constant uint& padded_batch,
    const constant uint& rows,
    const uint index AXIS(active_batch * rows, 256)) {
  const uint token = index / rows;
  const uint row = index - token * rows;
  output[index] = float(input[row * padded_batch + token]);
}

KERNEL(QtipD4S4EmbeddingLookup)(
    device const uint32_t* token_ids,
    device const uchar* codes,
    device const ushort* row_scales_bf16,
    device const uchar* ladder_indices,
    device const char* d4_table,
    device const half* ladder,
    device const int32_t* output_hadamard_factors,
    device bfloat* output,
    const constant uint& batch_size,
    const constant uint& vocab_size,
    const constant uint& model_dim,
    const constant float& input_scale,
    const uint block GROUPS(model_dim / 32u),
    const uint batch GROUPS(batch_size),
    const uint lane THREADS(32)) {
  const uint token = token_ids[batch];
  const uint column = block * 32u + lane;
  float value = 0.0f;
  if (token < vocab_size) {
    const uint code_stride = model_dim / 4u;
    const uint ladder_stride = model_dim / 128u;
    const uchar code = codes[token * code_stride + column / 4u];
    const int component = int(d4_table[uint(code) * 4u + (column & 3u)]);
    const uint scale_group = column / 64u;
    const uchar packed_ladder = ladder_indices[token * ladder_stride + scale_group / 2u];
    const uint ladder_index = (scale_group & 1u) == 0u ? uint(packed_ladder & 15u) : uint(packed_ladder >> 4u);
    const float row_scale = as_type<float>(uint(row_scales_bf16[token]) << 16u);
    value = (row_scale * float(ladder[ladder_index])) * float(component);
  }
  const bfloat folded = simdgroup_output_random_hadamard_transform(
      ushort(lane), bfloat(value), output_hadamard_factors[column]);
  output[batch * model_dim + column] = bfloat(float(folded) * input_scale);
}

KERNEL(QtipRht32Bf16Padded)(
    device const bfloat* input,
    device const int32_t* input_hadamard_factors,
    device bfloat* output,
    const constant uint& active_batch,
    const constant uint& padded_batch,
    const constant uint& dimension,
    const uint block GROUPS(padded_batch * (dimension / 32u)),
    const uint lane THREADS(32)) {
  const uint blocks_per_row = dimension / 32u;
  const uint token = block / blocks_per_row;
  const uint column = (block - token * blocks_per_row) * 32u + lane;
  float value = token < active_batch ? float(input[token * dimension + column]) : 0.0f;
  value = simdgroup_input_random_hadamard_transform(
      ushort(lane), value, input_hadamard_factors[column]);
  output[token * dimension + column] = bfloat(value);
}

template <ushort COLS, ushort SIMD_COLS, ushort THREAD_COUNT, ushort B_STRIDE>
static inline void qtip_i3_s4_readout_mxu(
    device const uchar* codes,
    device const bfloat* activations,
    device const ushort* row_scales_bf16,
    device const uchar* ladder_indices,
    device const half* ladder,
    device bfloat* output,
    uint rows,
    uint columns,
    uint code_stride,
    uint ladder_stride,
    threadgroup bfloat* a_shared,
    threadgroup bfloat* b_shared,
    uint row_tile,
    uint token_tile,
    uint thread_index,
    const ThreadContext thread_context) {
  using Tile = uzu::matmul::ThreadgroupTile<
      bfloat,
      bfloat,
      bfloat,
      QTIP_MMA_ROWS,
      COLS,
      QTIP_MMA_DEPTH,
      QTIP_MMA_SIMDGROUPS_ROWS,
      SIMD_COLS,
      false,
      false,
      QTIP_MMA_A_STRIDE,
      B_STRIDE,
      float>;

  const uint row_base = row_tile * QTIP_MMA_ROWS;
  const uint token_base = token_tile * COLS;
  thread Tile accumulator(thread_context);

  for (uint k_base = 0; k_base < columns; k_base += QTIP_MMA_DEPTH) {
    if (thread_index < QTIP_MMA_ROWS * 4u) {
      const uint local_row = thread_index >> 2u;
      const uint decode_lane = thread_index & 3u;
      const uint row = row_base + local_row;
      threadgroup bfloat* weight_row = a_shared + local_row * QTIP_MMA_A_STRIDE;
      if (row < rows) {
        device const uchar* row_codes = codes + row * code_stride;
        device const uchar* row_ladders = ladder_indices + row * ladder_stride;
        const float row_scale = as_type<float>(uint(row_scales_bf16[row]) << 16u);
        for (uint local_k = decode_lane; local_k < QTIP_MMA_DEPTH; local_k += 4u) {
          const uint column = k_base + local_k;
          const uint bit = column * 3u;
          const uint byte = bit >> 3u;
          const uint shift = bit & 7u;
          uint packed = uint(row_codes[byte]);
          if (shift > 5u) {
            packed |= uint(row_codes[byte + 1u]) << 8u;
          }
          const int level = int(((packed >> shift) & 7u) * 2u) - 7;
          const uint scale_group = column >> 6u;
          const uchar packed_ladder = row_ladders[scale_group >> 1u];
          const uint ladder_index = (scale_group & 1u) == 0u ? uint(packed_ladder & 15u) : uint(packed_ladder >> 4u);
          weight_row[local_k] = bfloat((row_scale * float(ladder[ladder_index])) * float(level));
        }
      } else {
        for (uint local_k = decode_lane; local_k < QTIP_MMA_DEPTH; local_k += 4u) {
          weight_row[local_k] = bfloat(0.0f);
        }
      }
    }

    for (uint index = thread_index; index < QTIP_MMA_DEPTH * COLS; index += THREAD_COUNT) {
      const uint local_k = index / COLS;
      const uint local_token = index - local_k * COLS;
      b_shared[local_k * B_STRIDE + local_token] =
          activations[(token_base + local_token) * columns + k_base + local_k];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    accumulator.matmul(a_shared, b_shared);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  accumulator.store_result(output + row_base * COLS + token_base, int(COLS));
}

#define QTIP_I3_S4_READOUT_MXU_KERNEL(NAME, COLS, SIMD_COLS, THREAD_COUNT, B_STRIDE) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const bfloat* activations, \
    device const ushort* row_scales_bf16, \
    device const uchar* ladder_indices, \
    device const half* ladder, \
    device bfloat* output, \
    const constant uint& rows, \
    const constant uint& columns, \
    const constant uint& code_stride, \
    const constant uint& ladder_stride, \
    threadgroup bfloat a_shared[QTIP_MMA_ROWS * QTIP_MMA_A_STRIDE], \
    threadgroup bfloat b_shared[QTIP_MMA_DEPTH * 66], \
    const uint row_tile GROUPS(rows.div_ceil(QTIP_MMA_ROWS)), \
    const uint token_tile GROUPS(1), \
    const uint thread_index THREADS(THREAD_COUNT), \
    const ThreadContext thread_context) { \
  qtip_i3_s4_readout_mxu<COLS, SIMD_COLS, THREAD_COUNT, B_STRIDE>( \
      codes, activations, row_scales_bf16, ladder_indices, ladder, output, \
      rows, columns, code_stride, ladder_stride, a_shared, b_shared, \
      row_tile, token_tile, thread_index, thread_context); \
}

QTIP_I3_S4_READOUT_MXU_KERNEL(QtipI3S4ReadoutMxuB16, 16, 2, 128, 18)
QTIP_I3_S4_READOUT_MXU_KERNEL(QtipI3S4ReadoutMxuB32, 32, 2, 128, 34)
QTIP_I3_S4_READOUT_MXU_KERNEL(QtipI3S4ReadoutMxuB64, 64, 4, 256, 66)

#undef QTIP_I3_S4_READOUT_MXU_KERNEL

// Sparse i3/S4 readout for the weaver: out[r][j] = <rht32(x_r), W[token_ids[r][j]]> with the weight
// reconstructed exactly as the dense MXU readout does (bf16-rounded row_scale * ladder * level).
// One threadgroup = (8 candidates, one row); the transformed activation row is staged once.
#define QTIP_I3_S4_SPARSE_MAX_COLUMNS 8192u
template <typename OutT>
static inline void qtip_i3_s4_readout_sparse(
    device const uchar* codes,
    device const bfloat* activations,
    device const ushort* row_scales_bf16,
    device const uchar* ladder_indices,
    device const half* ladder,
    device const uint* token_ids,
    device OutT* output,
    uint ids_per_row,
    uint columns,
    uint code_stride,
    uint ladder_stride,
    float soft_cap,
    threadgroup bfloat* a_shared,
    uint candidate_tile,
    uint row,
    uint thread_index) {
  for (uint c = thread_index; c < columns; c += 256u) {
    a_shared[c] = activations[row * columns + c];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint simd_id = thread_index >> 5u;
  const uint lane = thread_index & 31u;
  const uint candidate = candidate_tile * 8u + simd_id;
  if (candidate >= ids_per_row) {
    return;
  }
  const uint token = token_ids[row * ids_per_row + candidate];
  device const uchar* row_codes = codes + token * code_stride;
  device const uchar* row_ladders = ladder_indices + token * ladder_stride;
  const float row_scale = as_type<float>(uint(row_scales_bf16[token]) << 16u);
  float acc = 0.0f;
  for (uint column = lane * 8u; column < columns; column += 256u) {
    const uint byte = (column * 3u) >> 3u;
    const uint packed = uint(row_codes[byte]) | (uint(row_codes[byte + 1u]) << 8u) | (uint(row_codes[byte + 2u]) << 16u);
    const uint scale_group = column >> 6u;
    const uchar packed_ladder = row_ladders[scale_group >> 1u];
    const uint ladder_index = (scale_group & 1u) == 0u ? uint(packed_ladder & 15u) : uint(packed_ladder >> 4u);
    const float group_scale = row_scale * float(ladder[ladder_index]);
    for (uint i = 0u; i < 8u; ++i) {
      const int level = int(((packed >> (3u * i)) & 7u) * 2u) - 7;
      const bfloat weight = bfloat(group_scale * float(level));
      acc += float(weight) * float(a_shared[column + i]);
    }
  }
  acc = simd_sum(acc);
  if (lane == 0u) {
    if (soft_cap > 0.0f) {
      acc = soft_cap * metal::fast::tanh(acc / soft_cap);
    }
    output[row * ids_per_row + candidate] = OutT(acc);
  }
}

#define QTIP_I3_S4_READOUT_SPARSE_KERNEL(NAME, OUT_T) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const bfloat* activations, \
    device const ushort* row_scales_bf16, \
    device const uchar* ladder_indices, \
    device const half* ladder, \
    device const uint* token_ids, \
    device OUT_T* output, \
    const constant uint& rows, \
    const constant uint& ids_per_row, \
    const constant uint& columns, \
    const constant uint& code_stride, \
    const constant uint& ladder_stride, \
    const constant float& soft_cap, \
    threadgroup bfloat a_shared[QTIP_I3_S4_SPARSE_MAX_COLUMNS], \
    const uint candidate_tile GROUPS(ids_per_row.div_ceil(8)), \
    const uint row GROUPS(rows), \
    const uint thread_index THREADS(256)) { \
  qtip_i3_s4_readout_sparse<OUT_T>( \
      codes, activations, row_scales_bf16, ladder_indices, ladder, token_ids, output, \
      ids_per_row, columns, code_stride, ladder_stride, soft_cap, a_shared, \
      candidate_tile, row, thread_index); \
}

QTIP_I3_S4_READOUT_SPARSE_KERNEL(QtipI3S4ReadoutSparseBf16, bfloat)
QTIP_I3_S4_READOUT_SPARSE_KERNEL(QtipI3S4ReadoutSparseF32, float)

#undef QTIP_I3_S4_READOUT_SPARSE_KERNEL

// Hot-band merge for weaver residuals: rows with token id < hot_rows take the hot (higher precision) readout.
KERNEL(QtipResidualMergeHot)(
    device const bfloat* hot,
    device const bfloat* cold,
    device const uint* token_ids,
    device bfloat* output,
    const constant uint& hot_rows,
    const constant uint& count,
    const uint block GROUPS(count.div_ceil(256)),
    const uint lane THREADS(256)) {
  const uint index = block * 256u + lane;
  if (index >= count) {
    return;
  }
  output[index] = token_ids[index] < hot_rows ? hot[index] : cold[index];
}

#define QTIP_GAUSSIAN_XORSHIFT_ONE_MUL_V2_BODY(COLS, PARTITIONS, K2, K3, MIXER) \
  (void)thread_index; \
  (void)transition_bits; \
  qtip_gaussian_one_mul_a8_mxu_fused_split_k<COLS, COLS, PARTITIONS, K2, K3, MIXER>( \
      codes, activations, activation_scales, scales, gains, output, partials, \
      rows, groups, bytes_per_row, K2 ? 4u : 6u, row_tile, 0u, thread_context)

#undef QTIP_GAUSSIAN_XORSHIFT_ONE_MUL_V2_BODY


#define QTIP_GAUSSIAN_COMPUTED_K2_ROW_SPLIT_B64(NAME, ROW_SIMDGROUPS) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const half* gains, \
    device bfloat* output, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& transition_bits, \
    const uint row_tile GROUPS(rows.div_ceil(ROW_SIMDGROUPS * 16u)), \
    const uint thread_index THREADS(ROW_SIMDGROUPS * 32u), \
    const ThreadContext thread_context) { \
  (void)thread_index; \
  (void)transition_bits; \
  qtip_gaussian_one_mul_a8_mxu_paired_k3_row_split2<64, ROW_SIMDGROUPS, true>( \
      codes, activations, activation_scales, scales, gains, output, \
      rows, groups, bytes_per_row, row_tile, thread_context); \
}

QTIP_GAUSSIAN_COMPUTED_K2_ROW_SPLIT_B64(QtipGaussianComputedWalshOneMulA8MxuPairedK2RowSplit1B64, 1)
QTIP_GAUSSIAN_COMPUTED_K2_ROW_SPLIT_B64(QtipGaussianComputedWalshOneMulA8MxuPairedK2RowSplit2B64, 2)
QTIP_GAUSSIAN_COMPUTED_K2_ROW_SPLIT_B64(QtipGaussianComputedWalshOneMulA8MxuPairedK2RowSplit4B64, 4)
QTIP_GAUSSIAN_COMPUTED_K2_ROW_SPLIT_B64(QtipGaussianComputedWalshOneMulA8MxuPairedK2RowSplit8B64, 8)

#undef QTIP_GAUSSIAN_COMPUTED_K2_ROW_SPLIT_B64


#define QTIP_GAUSSIAN_ONE_MUL_A8_MXU_ATOMIC_SPLIT_BODY(COLS, PARTITIONS) \
  qtip_gaussian_one_mul_a8_mxu_atomic_split_k<COLS, PARTITIONS>( \
      codes, activations, activation_scales, scales, gains, output, sums, \
      rows, groups, bytes_per_row, transition_bits, row_tile, thread_index, thread_context)

#undef QTIP_GAUSSIAN_ONE_MUL_A8_MXU_ATOMIC_SPLIT_BODY


#undef QTIP_GAUSSIAN_FIXTURE_THREADS
