#include <metal_stdlib>
#include "../common/activation_quantization.h"
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/trellis.h"

using namespace metal;
namespace trellis = uzu::trellis;

#define TRANSFORM_THREADS 512

UZU_CONST uint TRANSFORM_SIMDGROUPS = TRANSFORM_THREADS / METAL_SIMD_SIZE;
UZU_CONST ushort LARGEST_LANE_STRIDE = METAL_SIMD_SIZE / 2;

UZU_CONST uint REGISTER_BUTTERFLY_WIDTH = 2048 / 1024;

UZU_CONST uint MAX_MIXING_DIMENSION = 17;
UZU_CONST uint NARROW_PASS_COLUMNS = 2;
UZU_CONST uint WIDE_PASS_COLUMNS = 4;
UZU_CONST uint WIDE_PASS_MIXING_DIMENSION_THRESHOLD = 8;

static METAL_FUNC float butterfly(float value, ushort lane, ushort lane_stride) {
  const float other = simd_shuffle_xor(value, lane_stride);
  return (lane & lane_stride) != 0 ? other - value : value + other;
}

// Transposed reads are METAL_SIMD_SIZE floats apart; one pad float per row spreads them across all banks.
static METAL_FUNC uint bank_conflict_free_offset(uint hadamard_index) {
  return hadamard_index + hadamard_index / METAL_SIMD_SIZE;
}

static METAL_FUNC uint hadamard_index_before_transpose(ushort lane, ushort simdgroup, uint value_index) {
  return METAL_SIMD_SIZE * (simdgroup + TRANSFORM_SIMDGROUPS * value_index) + lane;
}

template <uint HADAMARD_SIZE>
static METAL_FUNC uint hadamard_index_after_transpose(ushort lane, ushort simdgroup, uint value_index) {
  if (HADAMARD_SIZE == 1024) {
    return METAL_SIMD_SIZE * lane + simdgroup + TRANSFORM_SIMDGROUPS * value_index;
  }
  return METAL_SIMD_SIZE * (lane + METAL_SIMD_SIZE * (value_index % REGISTER_BUTTERFLY_WIDTH)) + simdgroup +
         TRANSFORM_SIMDGROUPS * (value_index / REGISTER_BUTTERFLY_WIDTH);
}

template <uint DIMENSION>
struct TransformPassLayout {
  static constant constexpr uint HADAMARD_SIZE = DIMENSION & (0u - DIMENSION);
  static constant constexpr uint MIXING_DIMENSION = DIMENSION / HADAMARD_SIZE;
  static constant constexpr uint VALUES_PER_THREAD = HADAMARD_SIZE / TRANSFORM_THREADS;
  // Four columns beat two at width 17408; width 5120 stays at two to keep its threadgroup memory.
  static constant constexpr uint COLUMNS_PER_PASS =
      HADAMARD_SIZE == 1024
          ? (MIXING_DIMENSION > WIDE_PASS_MIXING_DIMENSION_THRESHOLD ? WIDE_PASS_COLUMNS : NARROW_PASS_COLUMNS)
          : 1;
  static constant constexpr uint COLUMN_SCRATCH_SIZE = HADAMARD_SIZE + HADAMARD_SIZE / METAL_SIMD_SIZE;
  static constant constexpr uint SCRATCH_SIZE = COLUMNS_PER_PASS * COLUMN_SCRATCH_SIZE;
  typedef float SignFlippedInput[VALUES_PER_THREAD][MIXING_DIMENSION];
  typedef float RotatedColumns[MIXING_DIMENSION][VALUES_PER_THREAD];
};

template <typename Layout, uint COLUMNS>
static METAL_FUNC void hadamard_transform_lanes(
    thread float (&column_values)[COLUMNS][Layout::VALUES_PER_THREAD],
    const ushort lane
) {
  METAL_PRAGMA_UNROLL
  for (ushort lane_stride = 1; lane_stride <= LARGEST_LANE_STRIDE; lane_stride <<= 1) {
    METAL_PRAGMA_UNROLL
    for (uint column = 0; column < COLUMNS; ++column) {
      METAL_PRAGMA_UNROLL
      for (uint value_index = 0; value_index < Layout::VALUES_PER_THREAD; ++value_index) {
        column_values[column][value_index] = butterfly(column_values[column][value_index], lane, lane_stride);
      }
    }
  }
}

template <typename Layout, uint COLUMNS>
static METAL_FUNC void rotate_columns(
    const uint first_output_mixing_index,
    const thread typename Layout::SignFlippedInput& sign_flipped_input,
    const threadgroup float* shared_mixing,
    threadgroup float* scratch,
    thread typename Layout::RotatedColumns& rotated,
    thread float& local_maximum,
    const ushort lane,
    const ushort simdgroup
) {
  const float normalization = 1.0f / sqrt(float(Layout::HADAMARD_SIZE));

  float column_values[COLUMNS][Layout::VALUES_PER_THREAD] = {};
  METAL_PRAGMA_UNROLL
  for (uint mixing_index = 0; mixing_index < Layout::MIXING_DIMENSION; ++mixing_index) {
    METAL_PRAGMA_UNROLL
    for (uint column = 0; column < COLUMNS; ++column) {
      const float mixing =
          shared_mixing[(first_output_mixing_index + column) * Layout::MIXING_DIMENSION + mixing_index];
      METAL_PRAGMA_UNROLL
      for (uint value_index = 0; value_index < Layout::VALUES_PER_THREAD; ++value_index) {
        column_values[column][value_index] =
            fma(sign_flipped_input[value_index][mixing_index], mixing, column_values[column][value_index]);
      }
    }
  }
  hadamard_transform_lanes<Layout, COLUMNS>(column_values, lane);
  METAL_PRAGMA_UNROLL
  for (uint column = 0; column < COLUMNS; ++column) {
    METAL_PRAGMA_UNROLL
    for (uint value_index = 0; value_index < Layout::VALUES_PER_THREAD; ++value_index) {
      scratch
          [column * Layout::COLUMN_SCRATCH_SIZE +
           bank_conflict_free_offset(hadamard_index_before_transpose(lane, simdgroup, value_index))] =
              column_values[column][value_index];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  METAL_PRAGMA_UNROLL
  for (uint column = 0; column < COLUMNS; ++column) {
    METAL_PRAGMA_UNROLL
    for (uint value_index = 0; value_index < Layout::VALUES_PER_THREAD; ++value_index) {
      const uint hadamard_index = hadamard_index_after_transpose<Layout::HADAMARD_SIZE>(lane, simdgroup, value_index);
      column_values[column][value_index] =
          scratch[column * Layout::COLUMN_SCRATCH_SIZE + bank_conflict_free_offset(hadamard_index)];
    }
  }
  hadamard_transform_lanes<Layout, COLUMNS>(column_values, lane);
  if constexpr (Layout::HADAMARD_SIZE == 2048) {
    METAL_PRAGMA_UNROLL
    for (uint column = 0; column < COLUMNS; ++column) {
      METAL_PRAGMA_UNROLL
      for (uint value_index = 0; value_index < Layout::VALUES_PER_THREAD; value_index += REGISTER_BUTTERFLY_WIDTH) {
        const float lower = column_values[column][value_index];
        const float upper = column_values[column][value_index + 1];
        column_values[column][value_index] = lower + upper;
        column_values[column][value_index + 1] = lower - upper;
      }
    }
  }
  METAL_PRAGMA_UNROLL
  for (uint column = 0; column < COLUMNS; ++column) {
    METAL_PRAGMA_UNROLL
    for (uint value_index = 0; value_index < Layout::VALUES_PER_THREAD; ++value_index) {
      rotated[first_output_mixing_index + column][value_index] =
          float(bfloat(column_values[column][value_index] * normalization));
      local_maximum = max(local_maximum, abs(rotated[first_output_mixing_index + column][value_index]));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <uint DIMENSION>
VARIANTS(DIMENSION, 5120, 6144, 17408)
PUBLIC KERNEL(TrellisTransform)(
    device const bfloat* input,
    device const float* rht_factors,
    device const float* mixing,
    device int8_t* activations,
    device float4* column_group_sums,
    device float* scales,
    constant uint& batch,
    threadgroup float scratch[TransformPassLayout<DIMENSION>::SCRATCH_SIZE],
    threadgroup float simdgroup_maxima[TRANSFORM_SIMDGROUPS],
    threadgroup float shared_mixing[MAX_MIXING_DIMENSION * MAX_MIXING_DIMENSION],
    const uint token GROUPS(batch),
    const uint thread_index THREADS(TRANSFORM_THREADS),
    const ThreadContext thread_context
) {
  using Layout = TransformPassLayout<DIMENSION>;
  static_assert(Layout::HADAMARD_SIZE == 1024 || Layout::HADAMARD_SIZE == 2048, "unsupported Hadamard size");
  static_assert(Layout::MIXING_DIMENSION <= MAX_MIXING_DIMENSION, "mixing matrix does not fit threadgroup memory");

  const ushort lane = ushort(thread_context.simd_lane_id);
  const ushort simdgroup = ushort(thread_context.simdgroup_index);

  for (uint index = thread_index; index < Layout::MIXING_DIMENSION * Layout::MIXING_DIMENSION;
       index += TRANSFORM_THREADS) {
    shared_mixing[index] = mixing[index];
  }

  typename Layout::SignFlippedInput sign_flipped_input;
  METAL_PRAGMA_UNROLL
  for (uint value_index = 0; value_index < Layout::VALUES_PER_THREAD; ++value_index) {
    const uint hadamard_index = hadamard_index_before_transpose(lane, simdgroup, value_index);
    METAL_PRAGMA_UNROLL
    for (uint mixing_index = 0; mixing_index < Layout::MIXING_DIMENSION; ++mixing_index) {
      sign_flipped_input[value_index][mixing_index] =
          float(input[token * DIMENSION + hadamard_index * Layout::MIXING_DIMENSION + mixing_index]) *
          rht_factors[hadamard_index * Layout::MIXING_DIMENSION + mixing_index];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  typename Layout::RotatedColumns rotated;
  float local_maximum = 0.0f;
  uint output_mixing_index = 0;
  for (; output_mixing_index + Layout::COLUMNS_PER_PASS <= Layout::MIXING_DIMENSION;
       output_mixing_index += Layout::COLUMNS_PER_PASS) {
    rotate_columns<Layout, Layout::COLUMNS_PER_PASS>(
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
  if constexpr (Layout::MIXING_DIMENSION % Layout::COLUMNS_PER_PASS != 0) {
    rotate_columns<Layout, Layout::MIXING_DIMENSION % Layout::COLUMNS_PER_PASS>(
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

  const float maximum =
      reduce_activation_quantization_row_maximum<TRANSFORM_SIMDGROUPS>(local_maximum, simdgroup_maxima, thread_context);
  const float scale = int8_activation_scale(maximum);

  int local_column_group_sums[trellis::COLUMN_GROUP_COUNT] = {};
  METAL_PRAGMA_UNROLL
  for (uint mixing_index = 0; mixing_index < Layout::MIXING_DIMENSION; ++mixing_index) {
    METAL_PRAGMA_UNROLL
    for (uint value_index = 0; value_index < Layout::VALUES_PER_THREAD; ++value_index) {
      const uint hadamard_index = hadamard_index_after_transpose<Layout::HADAMARD_SIZE>(lane, simdgroup, value_index);
      const uint column = hadamard_index * Layout::MIXING_DIMENSION + mixing_index;
      const int8_t quantized = quantize_activation_int8(rotated[mixing_index][value_index], scale);
      activations[token * DIMENSION + column] = quantized;
      METAL_PRAGMA_UNROLL
      for (uint column_group_index = 0; column_group_index < trellis::COLUMN_GROUP_COUNT; ++column_group_index) {
        local_column_group_sums[column_group_index] +=
            (column % trellis::COLUMN_GROUP_COUNT) == column_group_index ? int(quantized) : 0;
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
    for (uint source_simdgroup = 0; source_simdgroup < TRANSFORM_SIMDGROUPS; ++source_simdgroup) {
      const uint first_group_sum = trellis::COLUMN_GROUP_COUNT * source_simdgroup;
      token_column_group_sums += float4(
          scratch[first_group_sum],
          scratch[first_group_sum + 1],
          scratch[first_group_sum + 2],
          scratch[first_group_sum + 3]
      );
    }
    column_group_sums[token] = token_column_group_sums;
    scales[token] = scale;
  }
}
