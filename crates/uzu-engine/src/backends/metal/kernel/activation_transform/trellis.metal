#include <metal_stdlib>
#include "../common/activation_quantization.h"
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/trellis.h"

using namespace metal;
namespace trellis = uzu::trellis;

#define THREADS_PER_THREADGROUP 512

UZU_CONST uint SIMDGROUPS_PER_THREADGROUP = THREADS_PER_THREADGROUP / METAL_SIMD_SIZE;

UZU_CONST uint THREADGROUP_MEMORY_LIMIT = 32768;

static METAL_FUNC float butterfly(float value, ushort lane, ushort lane_stride) {
  const float other = simd_shuffle_xor(value, lane_stride);
  return (lane & lane_stride) != 0 ? other - value : value + other;
}

static METAL_FUNC uint bank_conflict_free_offset(uint hadamard_index) {
  return hadamard_index + hadamard_index / METAL_SIMD_SIZE;
}

static METAL_FUNC uint hadamard_index_before_transpose(ushort lane, ushort simdgroup, uint value_index) {
  return METAL_SIMD_SIZE * (simdgroup + SIMDGROUPS_PER_THREADGROUP * value_index) + lane;
}

// Supported limits; arrays are sized for them.
UZU_CONST uint MAX_DIMENSION = 20480;
UZU_CONST uint MIN_HADAMARD_SIZE = 512;
UZU_CONST uint MAX_HADAMARD_SIZE = 4096;
UZU_CONST uint MAX_COLUMNS_PER_PASS = 4;

UZU_CONST uint MAX_MIXING_DIMENSION = MAX_DIMENSION / MIN_HADAMARD_SIZE;
UZU_CONST uint MAX_VALUES_PER_THREAD = MAX_HADAMARD_SIZE / THREADS_PER_THREADGROUP;
UZU_CONST uint MAX_PRIVATE_VALUES = MAX_DIMENSION / THREADS_PER_THREADGROUP;
UZU_CONST uint MAX_SCRATCH_SIZE = MAX_HADAMARD_SIZE + MAX_HADAMARD_SIZE / METAL_SIMD_SIZE;
UZU_CONST uint MAX_MIXING_SIZE = MAX_MIXING_DIMENSION * MAX_MIXING_DIMENSION;
static_assert(
    sizeof(float) * (MAX_SCRATCH_SIZE + SIMDGROUPS_PER_THREADGROUP + MAX_MIXING_SIZE) <= THREADGROUP_MEMORY_LIMIT,
    "does not fit threadgroup memory"
);

struct TransformLayout {
  uint hadamard_size;
  uint mixing_dimension;
  uint values_per_thread;
  uint in_thread_values;
  uint second_lane_stride_limit;
  uint columns_per_pass;
  uint column_scratch_size;

  METAL_FUNC TransformLayout(const uint dimension) {
    hadamard_size = dimension & (0u - dimension);
    mixing_dimension = dimension / hadamard_size;
    values_per_thread = hadamard_size / THREADS_PER_THREADGROUP;
    in_thread_values = hadamard_size > 1024 ? hadamard_size / 1024 : 1;
    second_lane_stride_limit = hadamard_size == THREADS_PER_THREADGROUP ? METAL_SIMD_SIZE / 4 : METAL_SIMD_SIZE / 2;
    columns_per_pass = hadamard_size >= 2048 ? 1 : (mixing_dimension > 8 ? 4 : 2);
    column_scratch_size = hadamard_size + hadamard_size / METAL_SIMD_SIZE;
  }
};

static METAL_FUNC uint
hadamard_index_after_transpose(const thread TransformLayout& layout, ushort lane, ushort simdgroup, uint value_index) {
  if (layout.hadamard_size == THREADS_PER_THREADGROUP) {
    constexpr uint half_simd = METAL_SIMD_SIZE / 2;
    return METAL_SIMD_SIZE * (lane % half_simd) + half_simd * (lane / half_simd) + simdgroup;
  } else {
    return METAL_SIMD_SIZE * (lane + METAL_SIMD_SIZE * (value_index % layout.in_thread_values)) + simdgroup +
           SIMDGROUPS_PER_THREADGROUP * (value_index / layout.in_thread_values);
  }
}

static METAL_FUNC void hadamard_transform_lanes(
    const thread TransformLayout& layout,
    const uint columns,
    thread float (&column_values)[MAX_COLUMNS_PER_PASS][MAX_VALUES_PER_THREAD],
    const ushort lane,
    const uint lane_stride_limit
) {
  METAL_PRAGMA_UNROLL
  for (ushort lane_stride = 1; lane_stride <= lane_stride_limit; lane_stride <<= 1) {
    METAL_PRAGMA_UNROLL
    for (uint column = 0; column < columns; ++column) {
      METAL_PRAGMA_UNROLL
      for (uint value_index = 0; value_index < layout.values_per_thread; ++value_index) {
        column_values[column][value_index] = butterfly(column_values[column][value_index], lane, lane_stride);
      }
    }
  }
}

static METAL_FUNC void rotate_columns(
    const thread TransformLayout& layout,
    const uint columns,
    const uint first_output_mixing_index,
    const thread float (&sign_flipped_input)[MAX_PRIVATE_VALUES],
    const threadgroup float* shared_mixing,
    threadgroup float* scratch,
    thread float (&rotated)[MAX_PRIVATE_VALUES],
    thread float& local_maximum,
    const ushort lane,
    const ushort simdgroup
) {
  const float normalization = 1.0f / sqrt(float(layout.hadamard_size));

  float column_values[MAX_COLUMNS_PER_PASS][MAX_VALUES_PER_THREAD] = {};
  METAL_PRAGMA_UNROLL
  for (uint mixing_index = 0; mixing_index < layout.mixing_dimension; ++mixing_index) {
    METAL_PRAGMA_UNROLL
    for (uint column = 0; column < columns; ++column) {
      const float mixing = shared_mixing[(first_output_mixing_index + column) * layout.mixing_dimension + mixing_index];
      METAL_PRAGMA_UNROLL
      for (uint value_index = 0; value_index < layout.values_per_thread; ++value_index) {
        column_values[column][value_index] =
            fma(sign_flipped_input[value_index * layout.mixing_dimension + mixing_index],
                mixing,
                column_values[column][value_index]);
      }
    }
  }
  hadamard_transform_lanes(layout, columns, column_values, lane, METAL_SIMD_SIZE / 2);
  METAL_PRAGMA_UNROLL
  for (uint column = 0; column < columns; ++column) {
    METAL_PRAGMA_UNROLL
    for (uint value_index = 0; value_index < layout.values_per_thread; ++value_index) {
      scratch
          [column * layout.column_scratch_size +
           bank_conflict_free_offset(hadamard_index_before_transpose(lane, simdgroup, value_index))] =
              column_values[column][value_index];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  METAL_PRAGMA_UNROLL
  for (uint column = 0; column < columns; ++column) {
    METAL_PRAGMA_UNROLL
    for (uint value_index = 0; value_index < layout.values_per_thread; ++value_index) {
      const uint hadamard_index = hadamard_index_after_transpose(layout, lane, simdgroup, value_index);
      column_values[column][value_index] =
          scratch[column * layout.column_scratch_size + bank_conflict_free_offset(hadamard_index)];
    }
  }
  hadamard_transform_lanes(layout, columns, column_values, lane, layout.second_lane_stride_limit);
  METAL_PRAGMA_UNROLL
  for (uint value_stride = 1; value_stride < layout.in_thread_values; value_stride <<= 1) {
    METAL_PRAGMA_UNROLL
    for (uint column = 0; column < columns; ++column) {
      METAL_PRAGMA_UNROLL
      for (uint block = 0; block < layout.values_per_thread; block += 2 * value_stride) {
        METAL_PRAGMA_UNROLL
        for (uint value_index = block; value_index < block + value_stride; ++value_index) {
          const float lower = column_values[column][value_index];
          const float upper = column_values[column][value_index + value_stride];
          column_values[column][value_index] = lower + upper;
          column_values[column][value_index + value_stride] = lower - upper;
        }
      }
    }
  }
  METAL_PRAGMA_UNROLL
  for (uint column = 0; column < columns; ++column) {
    METAL_PRAGMA_UNROLL
    for (uint value_index = 0; value_index < layout.values_per_thread; ++value_index) {
      thread float& output = rotated[(first_output_mixing_index + column) * layout.values_per_thread + value_index];
      output = float(bfloat(column_values[column][value_index] * normalization));
      local_maximum = max(local_maximum, abs(output));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

PUBLIC KERNEL(TrellisTransform)(
    device const bfloat* input,
    device const float* rht_factors,
    device const float* mixing,
    device int8_t* activations,
    device float4* column_group_sums,
    device float* scales,
    constant uint& batch,
    const uint dimension SPECIALIZE,
    threadgroup float scratch[MAX_SCRATCH_SIZE],
    threadgroup float simdgroup_maxima[SIMDGROUPS_PER_THREADGROUP],
    threadgroup float shared_mixing[MAX_MIXING_SIZE],
    const uint token GROUPS(batch),
    const uint thread_index THREADS(THREADS_PER_THREADGROUP),
    const ThreadContext thread_context
) {
  const TransformLayout layout(dimension);

  const ushort lane = ushort(thread_context.simd_lane_id);
  const ushort simdgroup = ushort(thread_context.simdgroup_index);

  for (uint index = thread_index; index < layout.mixing_dimension * layout.mixing_dimension;
       index += THREADS_PER_THREADGROUP) {
    shared_mixing[index] = mixing[index];
  }

  float sign_flipped_input[MAX_PRIVATE_VALUES];
  METAL_PRAGMA_UNROLL
  for (uint value_index = 0; value_index < layout.values_per_thread; ++value_index) {
    const uint hadamard_index = hadamard_index_before_transpose(lane, simdgroup, value_index);
    const uint first_column = hadamard_index * layout.mixing_dimension;
    const device bfloat* token_input = input + token * dimension + first_column;
    const device float* token_factors = rht_factors + first_column;
    METAL_PRAGMA_UNROLL
    for (uint mixing_index = 0; mixing_index < layout.mixing_dimension; ++mixing_index) {
      sign_flipped_input[value_index * layout.mixing_dimension + mixing_index] =
          float(token_input[mixing_index]) * token_factors[mixing_index];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float rotated[MAX_PRIVATE_VALUES];
  float local_maximum = 0.0f;
  uint output_mixing_index = 0;
  for (; output_mixing_index + layout.columns_per_pass <= layout.mixing_dimension;
       output_mixing_index += layout.columns_per_pass) {
    rotate_columns(
        layout,
        layout.columns_per_pass,
        output_mixing_index,
        sign_flipped_input,
        shared_mixing,
        scratch,
        rotated,
        local_maximum,
        lane,
        simdgroup
    );
  }
  if (layout.mixing_dimension % layout.columns_per_pass != 0) {
    rotate_columns(
        layout,
        layout.mixing_dimension % layout.columns_per_pass,
        output_mixing_index,
        sign_flipped_input,
        shared_mixing,
        scratch,
        rotated,
        local_maximum,
        lane,
        simdgroup
    );
  }

  const float maximum = reduce_activation_quantization_row_maximum<SIMDGROUPS_PER_THREADGROUP>(
      local_maximum,
      simdgroup_maxima,
      thread_context
  );
  const float scale = (isfinite(maximum) && maximum > 0.0f) ? precise::divide(maximum, 127.0f) : 1.0f;

  int local_column_group_sums[trellis::COLUMN_GROUP_COUNT] = {};
  METAL_PRAGMA_UNROLL
  for (uint value_index = 0; value_index < layout.values_per_thread; ++value_index) {
    const uint first_column =
        hadamard_index_after_transpose(layout, lane, simdgroup, value_index) * layout.mixing_dimension;
    device int8_t* token_activations = activations + token * dimension + first_column;
    METAL_PRAGMA_UNROLL
    for (uint mixing_index = 0; mixing_index < layout.mixing_dimension; ++mixing_index) {
      const int8_t quantized =
          quantize_activation_int8(rotated[mixing_index * layout.values_per_thread + value_index], scale);
      token_activations[mixing_index] = quantized;
      METAL_PRAGMA_UNROLL
      for (uint column_group_index = 0; column_group_index < trellis::COLUMN_GROUP_COUNT; ++column_group_index) {
        local_column_group_sums[column_group_index] +=
            ((first_column + mixing_index) % trellis::COLUMN_GROUP_COUNT) == column_group_index ? int(quantized) : 0;
      }
    }
  }
  METAL_PRAGMA_UNROLL
  for (uint column_group_index = 0; column_group_index < trellis::COLUMN_GROUP_COUNT; ++column_group_index) {
    const int simdgroup_sum = simd_sum(local_column_group_sums[column_group_index]);
    if (lane == 0) {
      scratch[trellis::COLUMN_GROUP_COUNT * simdgroup + column_group_index] = float(simdgroup_sum);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float4 token_column_group_sums = float4(0.0f);
    for (uint source_simdgroup = 0; source_simdgroup < SIMDGROUPS_PER_THREADGROUP; ++source_simdgroup) {
      const uint first_group_sum_index = trellis::COLUMN_GROUP_COUNT * source_simdgroup;
      token_column_group_sums += float4(
          scratch[first_group_sum_index],
          scratch[first_group_sum_index + 1],
          scratch[first_group_sum_index + 2],
          scratch[first_group_sum_index + 3]
      );
    }
    column_group_sums[token] = token_column_group_sums;
    scales[token] = scale;
  }
}
