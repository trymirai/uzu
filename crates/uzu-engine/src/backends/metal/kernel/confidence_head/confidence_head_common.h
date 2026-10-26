#pragma once

#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/defines.h"
#include "../common/thread_context.h"
#include "confidence_head_layout.h"

using namespace metal;

// Threads of the head's threadgroups (attention, tree tokens, row projections).
#define HEAD_THREADS 256u
#define HEAD_SIMDGROUPS (HEAD_THREADS / METAL_SIMD_SIZE)

#define CANDIDATES_MAX 512u
#define CH_HIDDEN_MAX 64u
#define CH_CHILDREN_MAX 4u
#define CH_EDGE_HIDDEN_MAX 128u
#define CH_EDGE_INPUT_MAX (3u * CH_HIDDEN_MAX + CH_SLOT)
#define CH_NONE 0xffffffffu

// Several sums (or maxima) over a simdgroup's lanes at once, by a butterfly: each step halves the values a lane holds,
// so eight values take 4 + 2 + 1 + 1 + 1 shuffles and four take 2 + 1 + 1 + 1 + 1, instead of five per value.
// With eight values lane l gets value (l >> 2) & 7, with four value (l >> 3) & 3.
// The lanes that differ from l only in the lower bits hold the same result.
template <bool MAXIMUM>
METAL_FUNC float confidence_head_combine(float a, float b) {
  return MAXIMUM ? max(a, b) : a + b;
}

template <bool MAXIMUM>
METAL_FUNC float confidence_head_reduce_low(float value, uint from) {
  for (ushort mask = ushort(from); mask > 0; mask >>= 1) {
    value = confidence_head_combine<MAXIMUM>(value, simd_shuffle_xor(value, mask));
  }
  return value;
}

template <bool MAXIMUM>
METAL_FUNC float confidence_head_reduce8(thread const float* values, uint lane) {
  const bool high16 = (lane & 16u) != 0u, high8 = (lane & 8u) != 0u, high4 = (lane & 4u) != 0u;
  float four[4];
  METAL_PRAGMA_UNROLL
  for (uint index = 0; index < 4u; ++index) {
    const float keep = high16 ? values[4u + index] : values[index];
    const float give = high16 ? values[index] : values[4u + index];
    four[index] = confidence_head_combine<MAXIMUM>(keep, simd_shuffle_xor(give, ushort(16)));
  }
  float two[2];
  METAL_PRAGMA_UNROLL
  for (uint index = 0; index < 2u; ++index) {
    const float keep = high8 ? four[2u + index] : four[index];
    const float give = high8 ? four[index] : four[2u + index];
    two[index] = confidence_head_combine<MAXIMUM>(keep, simd_shuffle_xor(give, ushort(8)));
  }
  const float one =
      confidence_head_combine<MAXIMUM>(high4 ? two[1] : two[0], simd_shuffle_xor(high4 ? two[0] : two[1], ushort(4)));
  return confidence_head_reduce_low<MAXIMUM>(one, 2u);
}

METAL_FUNC uint confidence_head_reduce8_index(uint lane) { return (lane >> 2) & 7u; }

template <bool MAXIMUM>
METAL_FUNC float confidence_head_reduce4(float4 values, uint lane) {
  const bool high16 = (lane & 16u) != 0u, high8 = (lane & 8u) != 0u;
  const float2 keep = high16 ? values.zw : values.xy, give = high16 ? values.xy : values.zw;
  const float2 two = float2(
      confidence_head_combine<MAXIMUM>(keep.x, simd_shuffle_xor(give.x, ushort(16))),
      confidence_head_combine<MAXIMUM>(keep.y, simd_shuffle_xor(give.y, ushort(16)))
  );
  const float one =
      confidence_head_combine<MAXIMUM>(high8 ? two.y : two.x, simd_shuffle_xor(high8 ? two.x : two.y, ushort(8)));
  return confidence_head_reduce_low<MAXIMUM>(one, 4u);
}

METAL_FUNC uint confidence_head_reduce4_index(uint lane) { return (lane >> 3) & 3u; }

// Sum and maximum over a threadgroup of SIMDGROUPS simdgroups (at most one per lane); every thread gets the result. The
// simdgroups' partials meet in threadgroup memory, and every simdgroup reduces them over its lanes, lane g reading
// simdgroup g's.
template <uint SIMDGROUPS = HEAD_SIMDGROUPS>
METAL_FUNC float confidence_head_sum(float value, threadgroup float* shared, const thread ThreadContext& context) {
  const float partial = simd_sum(value);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (context.simd_lane_id == 0) {
    shared[context.simdgroup_index] = partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  static_assert(SIMDGROUPS <= METAL_SIMD_SIZE, "a lane per simdgroup");
  return simd_sum(context.simd_lane_id < SIMDGROUPS ? shared[context.simd_lane_id] : 0.0f);
}

template <uint SIMDGROUPS = HEAD_SIMDGROUPS>
METAL_FUNC float confidence_head_max(float value, threadgroup float* shared, const thread ThreadContext& context) {
  const float partial = simd_max(value);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (context.simd_lane_id == 0) {
    shared[context.simdgroup_index] = partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  static_assert(SIMDGROUPS <= METAL_SIMD_SIZE, "a lane per simdgroup");
  return simd_max(context.simd_lane_id < SIMDGROUPS ? shared[context.simd_lane_id] : -INFINITY);
}

// log-sum-exp over a threadgroup in one exchange: each simdgroup's maximum and its sum of exp(x - that maximum) meet in
// threadgroup memory and are merged over the lanes (lane g reads simdgroup g's pair). `counted` says whether this
// thread's value enters the sum; the maximum is over all values. Returns (maximum, sum of exp(x - maximum)). One value
// a thread (shared holds 2 x SIMDGROUPS floats), or two at once (shared holds SIMDGROUPS float4).
template <uint SIMDGROUPS>
METAL_FUNC float2
confidence_head_log_sum_exp(float value, bool counted, threadgroup float* shared, const thread ThreadContext& context) {
  static_assert(SIMDGROUPS <= METAL_SIMD_SIZE, "a lane per simdgroup");
  const uint lane = context.simd_lane_id;
  const float top = simd_max(value);
  const float total = simd_sum(counted ? exp(value - top) : 0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    shared[2u * context.simdgroup_index] = top;
    shared[2u * context.simdgroup_index + 1u] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float group_top = lane < SIMDGROUPS ? shared[2u * lane] : -INFINITY;
  const float group_total = lane < SIMDGROUPS ? shared[2u * lane + 1u] : 0.0f;
  const float maximum = simd_max(group_top);
  return float2(maximum, simd_sum(group_total > 0.0f ? group_total * exp(group_top - maximum) : 0.0f));
}

static_assert(CH_CHILDREN_MAX == 4u, "the children's sums travel as one float4");

// Four sums, or four maxima, over the local part's threadgroup at once; every thread gets them.
template <uint SIMDGROUPS>
METAL_FUNC float4 confidence_head_sum4(float4 value, threadgroup float4* shared, const thread ThreadContext& context) {
  const uint lane = context.simd_lane_id;
  const float partial = confidence_head_reduce4<false>(value, lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((lane & 7u) == 0u) {
    reinterpret_cast<threadgroup float*>(shared)[context.simdgroup_index * 4u + confidence_head_reduce4_index(lane)] =
        partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  static_assert(SIMDGROUPS <= METAL_SIMD_SIZE, "a lane per simdgroup");
  return simd_sum(lane < SIMDGROUPS ? shared[lane] : float4(0.0f));
}

template <uint SIMDGROUPS>
METAL_FUNC float4 confidence_head_max4(float4 value, threadgroup float4* shared, const thread ThreadContext& context) {
  const uint lane = context.simd_lane_id;
  const float partial = confidence_head_reduce4<true>(value, lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((lane & 7u) == 0u) {
    reinterpret_cast<threadgroup float*>(shared)[context.simdgroup_index * 4u + confidence_head_reduce4_index(lane)] =
        partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  static_assert(SIMDGROUPS <= METAL_SIMD_SIZE, "a lane per simdgroup");
  return simd_max(lane < SIMDGROUPS ? shared[lane] : float4(-INFINITY));
}

// Units per simdgroup in a dense layer. A lane reads four adjacent columns at once, one float4 load each (every
// layer's width is a multiple of four: the head checks it when it loads).
#define CH_DENSE_UNITS 2u

// What a dense layer does with an output: store it, store its SiLU, or add it onto what the output holds (a residual).
// Fusing the last two into the layer saves a pass over the outputs and a barrier each.
enum class ConfidenceHeadEpilogue { Store, Silu, Accumulate };

// A dense layer on up to CH_CHILDREN_MAX rows at once, outputs[r * output_stride + u] = biases[u] + weights[u] .
// inputs[r * input_stride ...]: each simdgroup takes blocks of CH_DENSE_UNITS units, its lanes split the columns (so
// the weights are read in order, coalesced) and every (unit, row) dot product ends in one simd_sum. All threads of the
// threadgroup call it; the caller puts a barrier before reading the outputs.
template <uint THREADS, ConfidenceHeadEpilogue EPILOGUE = ConfidenceHeadEpilogue::Store, uint QUADS = 1u>
METAL_FUNC void confidence_head_dense_rows(
    const device float* weights,
    const device float* biases,
    uint units,
    uint columns,
    threadgroup const float* inputs,
    uint input_stride,
    uint rows,
    threadgroup float* outputs,
    uint output_stride,
    const thread ThreadContext& context
) {
  const uint lane = context.simd_lane_id;
  for (uint first = context.simdgroup_index * CH_DENSE_UNITS; first < units;
       first += (THREADS / METAL_SIMD_SIZE) * CH_DENSE_UNITS) {
    float partial[CH_DENSE_UNITS][CH_CHILDREN_MAX];
    METAL_PRAGMA_UNROLL
    for (uint unit = 0; unit < CH_DENSE_UNITS; ++unit) {
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < CH_CHILDREN_MAX; ++row) {
        partial[unit][row] = 0.0f;
      }
    }
    // QUADS float4 loads a lane per unit and row and per step of 4 x 32 x QUADS columns.
    for (uint start = 4u * lane; start < columns; start += 4u * METAL_SIMD_SIZE * QUADS) {
      float4 weight[QUADS][CH_DENSE_UNITS];
      float4 input[QUADS][CH_CHILDREN_MAX];
      METAL_PRAGMA_UNROLL
      for (uint step = 0; step < QUADS; ++step) {
        const uint column = start + step * 4u * METAL_SIMD_SIZE;
        const bool inside = column < columns;
        METAL_PRAGMA_UNROLL
        for (uint unit = 0; unit < CH_DENSE_UNITS; ++unit) {
          weight[step][unit] =
              inside && first + unit < units
                  ? float4(*reinterpret_cast<const device packed_float4*>(weights + (first + unit) * columns + column))
                  : float4(0.0f);
        }
        METAL_PRAGMA_UNROLL
        for (uint row = 0; row < CH_CHILDREN_MAX; ++row) {
          input[step][row] =
              inside && row < rows
                  ? float4(*reinterpret_cast<const threadgroup packed_float4*>(inputs + row * input_stride + column))
                  : float4(0.0f);
        }
      }
      METAL_PRAGMA_UNROLL
      for (uint step = 0; step < QUADS; ++step) {
        METAL_PRAGMA_UNROLL
        for (uint unit = 0; unit < CH_DENSE_UNITS; ++unit) {
          METAL_PRAGMA_UNROLL
          for (uint row = 0; row < CH_CHILDREN_MAX; ++row) {
            partial[unit][row] += dot(weight[step][unit], input[step][row]);
          }
        }
      }
    }
    // All CH_DENSE_UNITS x CH_CHILDREN_MAX sums over the lanes in one butterfly.
    static_assert(CH_DENSE_UNITS * CH_CHILDREN_MAX == 8u, "the butterfly reduces eight values");
    float values[8];
    METAL_PRAGMA_UNROLL
    for (uint index = 0; index < 8u; ++index) {
      values[index] = partial[index / CH_CHILDREN_MAX][index % CH_CHILDREN_MAX];
    }
    const float total = confidence_head_reduce8<false>(values, lane);
    const uint index = confidence_head_reduce8_index(lane);
    const uint unit = index / CH_CHILDREN_MAX, row = index % CH_CHILDREN_MAX;
    if ((lane & 3u) == 0u) {
      {
        if (first + unit < units && row < rows && EPILOGUE == ConfidenceHeadEpilogue::Store) {
          outputs[row * output_stride + first + unit] = total + biases[first + unit];
        }
        if (first + unit < units && row < rows && EPILOGUE != ConfidenceHeadEpilogue::Store) {
          const float value = total + biases[first + unit];
          threadgroup float* output = outputs + row * output_stride + first + unit;
          *output = EPILOGUE == ConfidenceHeadEpilogue::Silu ? activate_silu(value) : *output + value;
        }
      }
    }
  }
}

// A dense layer of few inputs (up to 64 a pass): each half of a simdgroup takes CH_DENSE_UNITS units and its 16 lanes
// read four adjacent inputs each, so no lane idles on a 64-wide input; a half's eight sums reduce over its 16 lanes.
template <uint THREADS, ConfidenceHeadEpilogue EPILOGUE>
METAL_FUNC void confidence_head_dense_rows_narrow(
    const device float* weights,
    const device float* biases,
    uint units,
    uint columns,
    threadgroup const float* inputs,
    uint input_stride,
    uint rows,
    threadgroup float* outputs,
    uint output_stride,
    const thread ThreadContext& context
) {
  static_assert(CH_DENSE_UNITS * CH_CHILDREN_MAX == 8u, "a half reduces eight values");
  const uint lane = context.simd_lane_id, half_lane = lane % 16u, half_index = lane / 16u;
  for (uint block = context.simdgroup_index * 2u * CH_DENSE_UNITS; block < units;
       block += (THREADS / METAL_SIMD_SIZE) * 2u * CH_DENSE_UNITS) {
    const uint first = block + half_index * CH_DENSE_UNITS;
    float values[8];
    METAL_PRAGMA_UNROLL
    for (uint index = 0; index < 8u; ++index) {
      values[index] = 0.0f;
    }
    for (uint column = 4u * half_lane; column < columns; column += 64u) {
      float4 weight[CH_DENSE_UNITS];
      METAL_PRAGMA_UNROLL
      for (uint unit = 0; unit < CH_DENSE_UNITS; ++unit) {
        weight[unit] =
            first + unit < units
                ? float4(*reinterpret_cast<const device packed_float4*>(weights + (first + unit) * columns + column))
                : float4(0.0f);
      }
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < CH_CHILDREN_MAX; ++row) {
        const float4 input =
            row < rows
                ? float4(*reinterpret_cast<const threadgroup packed_float4*>(inputs + row * input_stride + column))
                : float4(0.0f);
        METAL_PRAGMA_UNROLL
        for (uint unit = 0; unit < CH_DENSE_UNITS; ++unit) {
          values[unit * CH_CHILDREN_MAX + row] += dot(weight[unit], input);
        }
      }
    }
    const bool high8 = (lane & 8u) != 0u, high4 = (lane & 4u) != 0u, high2 = (lane & 2u) != 0u;
    float four[4];
    METAL_PRAGMA_UNROLL
    for (uint index = 0; index < 4u; ++index) {
      const float keep = high8 ? values[4u + index] : values[index];
      const float give = high8 ? values[index] : values[4u + index];
      four[index] = keep + simd_shuffle_xor(give, ushort(8));
    }
    float two[2];
    METAL_PRAGMA_UNROLL
    for (uint index = 0; index < 2u; ++index) {
      const float keep = high4 ? four[2u + index] : four[index];
      const float give = high4 ? four[index] : four[2u + index];
      two[index] = keep + simd_shuffle_xor(give, ushort(4));
    }
    float total = (high2 ? two[1] : two[0]) + simd_shuffle_xor(high2 ? two[0] : two[1], ushort(2));
    total += simd_shuffle_xor(total, ushort(1));
    const uint index = (half_lane >> 1) & 7u;
    const uint unit = index / CH_CHILDREN_MAX, row = index % CH_CHILDREN_MAX;
    if ((lane & 1u) == 0u && first + unit < units && row < rows) {
      const float value = total + biases[first + unit];
      threadgroup float* output = outputs + row * output_stride + first + unit;
      *output = EPILOGUE == ConfidenceHeadEpilogue::Store  ? value
                : EPILOGUE == ConfidenceHeadEpilogue::Silu ? activate_silu(value)
                                                           : *output + value;
    }
  }
}

// Layer norm of `rows` rows of `count` values from `source` (row r at source + r * source_stride) into `target`, one
// simdgroup per row: mean, then the mean of squared deviations. The rows may be the same memory (in place): each lane
// reads only what it writes. Ends with a barrier.
METAL_FUNC void confidence_head_layer_norm_rows_into(
    threadgroup const float* source,
    uint source_stride,
    threadgroup float* target,
    uint target_stride,
    uint rows,
    uint count,
    const device float* scales,
    const device float* biases,
    float epsilon,
    const thread ThreadContext& context
) {
  const uint row = context.simdgroup_index, lane = context.simd_lane_id;
  if (row < rows) {
    threadgroup const float* line = source + row * source_stride;
    threadgroup float* out = target + row * target_stride;
    float sum = 0.0f;
    for (uint index = lane; index < count; index += METAL_SIMD_SIZE) {
      sum += line[index];
    }
    const float mean = simd_sum(sum) / float(count);
    float squares = 0.0f;
    for (uint index = lane; index < count; index += METAL_SIMD_SIZE) {
      const float centred = line[index] - mean;
      squares += centred * centred;
    }
    const float inverse = 1.0f / sqrt(simd_sum(squares) / float(count) + epsilon);
    for (uint index = lane; index < count; index += METAL_SIMD_SIZE) {
      out[index] = (line[index] - mean) * inverse * scales[index] + biases[index];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

METAL_FUNC void confidence_head_layer_norm_rows(
    threadgroup float* values,
    uint stride,
    uint rows,
    uint count,
    const device float* scales,
    const device float* biases,
    float epsilon,
    const thread ThreadContext& context
) {
  confidence_head_layer_norm_rows_into(values, stride, values, stride, rows, count, scales, biases, epsilon, context);
}

// alpha and beta from z_u, into scalars[0] and scalars[1]: simdgroup o computes output o and its lane 0 applies the
// capped softplus. Ends with a barrier.
template <uint THREADS>
METAL_FUNC void confidence_head_alpha_beta(
    threadgroup const float* node,
    const device float* weights,
    const thread ConfidenceHeadLayout& layout,
    uint hidden,
    float cap,
    threadgroup float* scalars,
    const thread ThreadContext& context,
    uint lid
) {
  static_assert(THREADS >= 2u * METAL_SIMD_SIZE, "alpha and beta take a simdgroup each");
  (void)lid;
  const uint output = context.simdgroup_index, lane = context.simd_lane_id;
  if (output < 2u) {
    const device float* row = weights + layout.output_weights + output * hidden;
    float partial = 0.0f;
    for (uint column = lane; column < hidden; column += METAL_SIMD_SIZE) {
      partial = fma(row[column], node[column], partial);
    }
    const float total = simd_sum(partial);
    if (lane == 0u) {
      scalars[output] = min(activate_softplus(total + weights[layout.output_biases + output]), cap);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// One child's row of the edge MLP's input, [z_u; e_c; z_u * e_c; slot], normalised by the input layer norm into
// edge_in; lanes take the columns. Called by one simdgroup per child.
METAL_FUNC void confidence_head_edge_input_row(
    uint child,
    threadgroup const float* node,
    threadgroup const float* vectors,
    threadgroup const float* numbers,
    threadgroup float* edge_in,
    uint hidden,
    const device float* scales,
    const device float* biases,
    float epsilon,
    uint lane
) {
  const uint width = confidence_head_edge_input(hidden);
  threadgroup float* line = edge_in + child * CH_EDGE_INPUT_MAX;
  float sum = 0.0f;
  for (uint column = lane; column < width; column += METAL_SIMD_SIZE) {
    float value;
    if (column < hidden) {
      value = node[column];
    } else if (column < 2u * hidden) {
      value = vectors[child * hidden + column - hidden];
    } else if (column < 3u * hidden) {
      value = node[column - 2u * hidden] * vectors[child * hidden + column - 2u * hidden];
    } else {
      value = numbers[child * CH_SLOT + column - 3u * hidden];
    }
    line[column] = value;
    sum += value;
  }
  const float mean = simd_sum(sum) / float(width);
  float squares = 0.0f;
  for (uint column = lane; column < width; column += METAL_SIMD_SIZE) {
    const float centred = line[column] - mean;
    squares += centred * centred;
  }
  const float inverse = 1.0f / sqrt(simd_sum(squares) / float(width) + epsilon);
  for (uint column = lane; column < width; column += METAL_SIMD_SIZE) {
    line[column] = (line[column] - mean) * inverse * scales[column] + biases[column];
  }
}

// The final layer norm of one child's block state and delta = w . that + b on it, by one simdgroup.
METAL_FUNC void confidence_head_final_delta_row(
    uint child,
    threadgroup const float* edge_state,
    threadgroup float* deltas,
    const device float* weights,
    const thread ConfidenceHeadLayout& layout,
    uint hidden,
    float epsilon,
    uint lane
) {
  threadgroup const float* line = edge_state + child * CH_HIDDEN_MAX;
  float sum = 0.0f;
  for (uint index = lane; index < hidden; index += METAL_SIMD_SIZE) {
    sum += line[index];
  }
  const float mean = simd_sum(sum) / float(hidden);
  float squares = 0.0f;
  for (uint index = lane; index < hidden; index += METAL_SIMD_SIZE) {
    const float centred = line[index] - mean;
    squares += centred * centred;
  }
  const float inverse = 1.0f / sqrt(simd_sum(squares) / float(hidden) + epsilon);
  float partial = 0.0f;
  for (uint index = lane; index < hidden; index += METAL_SIMD_SIZE) {
    const float normed = (line[index] - mean) * inverse * weights[layout.edge_final_scales + index] +
                         weights[layout.edge_final_biases + index];
    partial = fma(weights[layout.delta_weights + index], normed, partial);
  }
  const float total = simd_sum(partial);
  if (lane == 0u) {
    deltas[child] = total + weights[layout.delta_bias];
  }
}

// delta of every child (deltas[c]) from z_u (node), the children's token vectors (vectors, C x H) and slot numbers
// (numbers, C x 4): the residual edge MLP of depth one over [z_u; e_c; z_u * e_c; slot]. The input is built inside its
// layer norm, the SiLU and the residual go with the dense layers' writes, and the last layer norm goes with delta.
// Ends with a barrier.
template <uint THREADS>
METAL_FUNC void confidence_head_edge_deltas(
    threadgroup const float* node,
    threadgroup const float* vectors,
    threadgroup const float* numbers,
    threadgroup float* edge_in,
    threadgroup float* edge_state,
    threadgroup float* edge_raised,
    threadgroup float* deltas,
    const device float* weights,
    const thread ConfidenceHeadLayout& layout,
    uint children,
    uint hidden,
    uint edge_hidden,
    float epsilon,
    const thread ThreadContext& context,
    uint lid
) {
  static_assert(THREADS >= CH_CHILDREN_MAX * METAL_SIMD_SIZE, "a simdgroup per child");
  (void)lid;
  const uint width = confidence_head_edge_input(hidden);
  const uint child = context.simdgroup_index, lane = context.simd_lane_id;
  if (child < children) {
    confidence_head_edge_input_row(
        child,
        node,
        vectors,
        numbers,
        edge_in,
        hidden,
        weights + layout.edge_input_scales,
        weights + layout.edge_input_biases,
        epsilon,
        lane
    );
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The block's state (kept for the residual), then its normed copy, the up layer's input.
  confidence_head_dense_rows<THREADS, ConfidenceHeadEpilogue::Store, 2u>(
      weights + layout.edge_entry_weights,
      weights + layout.edge_entry_biases,
      hidden,
      width,
      edge_in,
      CH_EDGE_INPUT_MAX,
      children,
      edge_state,
      CH_HIDDEN_MAX,
      context
  );
  threadgroup_barrier(mem_flags::mem_threadgroup);
  confidence_head_layer_norm_rows_into(
      edge_state,
      CH_HIDDEN_MAX,
      edge_in,
      CH_EDGE_INPUT_MAX,
      children,
      hidden,
      weights + layout.edge_block_scales,
      weights + layout.edge_block_biases,
      epsilon,
      context
  );
  confidence_head_dense_rows_narrow<THREADS, ConfidenceHeadEpilogue::Silu>(
      weights + layout.edge_up_weights,
      weights + layout.edge_up_biases,
      edge_hidden,
      hidden,
      edge_in,
      CH_EDGE_INPUT_MAX,
      children,
      edge_raised,
      CH_EDGE_HIDDEN_MAX,
      context
  );
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The down layer's outputs onto the block's state.
  confidence_head_dense_rows<THREADS, ConfidenceHeadEpilogue::Accumulate>(
      weights + layout.edge_down_weights,
      weights + layout.edge_down_biases,
      hidden,
      edge_hidden,
      edge_raised,
      CH_EDGE_HIDDEN_MAX,
      children,
      edge_state,
      CH_HIDDEN_MAX,
      context
  );
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (child < children) {
    confidence_head_final_delta_row(child, edge_state, deltas, weights, layout, hidden, epsilon, lane);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// min(log q_u(c), 0) of every child into edges[c]: scores alpha * l + beta * G over the pool (this thread's candidate
// `index`, counted when `valid`), delta added on the real children's indices, minus their log-sum-exp (shared holds
// 2 x THREADS / simdgroup size floats).
template <uint THREADS>
METAL_FUNC void confidence_head_child_edges(
    float logit,
    float gumbel,
    bool valid,
    uint index,
    threadgroup const uint* indices,
    threadgroup const float* deltas,
    threadgroup const uint* real,
    threadgroup float* edges,
    threadgroup float* shared,
    threadgroup float* scalars,
    uint children,
    const thread ThreadContext& context
) {
  const float alpha = scalars[0], beta = scalars[1];
  float score = valid ? alpha * logit + beta * gumbel : -INFINITY;
  for (uint child = 0; child < children; ++child) {
    if (valid && real[child] != 0u && indices[child] == index) {
      score += deltas[child];
    }
  }
  const float2 pool = confidence_head_log_sum_exp<THREADS / METAL_SIMD_SIZE>(score, valid, shared, context);
  const float normaliser = pool.x + log(pool.y);
  for (uint child = 0; child < children; ++child) {
    if (valid && indices[child] == index) {
      edges[child] = min(score - normaliser, 0.0f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
