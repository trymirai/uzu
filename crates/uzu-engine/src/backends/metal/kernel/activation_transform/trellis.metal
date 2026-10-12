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

template <uint DIMENSION>
struct TransformLayout {
  static constant constexpr uint HADAMARD_SIZE = DIMENSION & (0u - DIMENSION);
  static constant constexpr uint MIXING_DIMENSION = DIMENSION / HADAMARD_SIZE;
  static constant constexpr uint VALUES_PER_THREAD = HADAMARD_SIZE / THREADS_PER_THREADGROUP;
  static constant constexpr uint IN_THREAD_VALUES = HADAMARD_SIZE > 1024 ? HADAMARD_SIZE / 1024 : 1;
  static constant constexpr uint SECOND_LANE_STRIDE_LIMIT =
      HADAMARD_SIZE == THREADS_PER_THREADGROUP ? METAL_SIMD_SIZE / 4 : METAL_SIMD_SIZE / 2;
  static constant constexpr uint COLUMNS_PER_PASS = HADAMARD_SIZE >= 2048 ? 1 : (MIXING_DIMENSION > 8 ? 4 : 2);
  static constant constexpr uint COLUMN_SCRATCH_SIZE = HADAMARD_SIZE + HADAMARD_SIZE / METAL_SIMD_SIZE;
  static constant constexpr uint SCRATCH_SIZE = COLUMNS_PER_PASS * COLUMN_SCRATCH_SIZE;
  static constant constexpr uint MIXING_SIZE = MIXING_DIMENSION * MIXING_DIMENSION;
  static constant constexpr uint THREADGROUP_BYTES =
      sizeof(float) * (SCRATCH_SIZE + SIMDGROUPS_PER_THREADGROUP + MIXING_SIZE);
  typedef float SignFlippedInput[VALUES_PER_THREAD][MIXING_DIMENSION];
  typedef float RotatedColumns[MIXING_DIMENSION][VALUES_PER_THREAD];
};

template <typename Layout>
static METAL_FUNC uint hadamard_index_after_transpose(ushort lane, ushort simdgroup, uint value_index) {
  if constexpr (Layout::HADAMARD_SIZE == THREADS_PER_THREADGROUP) {
    constexpr uint half_simd = METAL_SIMD_SIZE / 2;
    return METAL_SIMD_SIZE * (lane % half_simd) + half_simd * (lane / half_simd) + simdgroup;
  } else {
    return METAL_SIMD_SIZE * (lane + METAL_SIMD_SIZE * (value_index % Layout::IN_THREAD_VALUES)) + simdgroup +
           SIMDGROUPS_PER_THREADGROUP * (value_index / Layout::IN_THREAD_VALUES);
  }
}

template <typename Layout, uint COLUMNS, uint LANE_STRIDE_LIMIT = METAL_SIMD_SIZE / 2>
static METAL_FUNC void hadamard_transform_lanes(
    thread float (&column_values)[COLUMNS][Layout::VALUES_PER_THREAD],
    const ushort lane
) {
  METAL_PRAGMA_UNROLL
  for (ushort lane_stride = 1; lane_stride <= LANE_STRIDE_LIMIT; lane_stride <<= 1) {
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
      const uint hadamard_index = hadamard_index_after_transpose<Layout>(lane, simdgroup, value_index);
      column_values[column][value_index] =
          scratch[column * Layout::COLUMN_SCRATCH_SIZE + bank_conflict_free_offset(hadamard_index)];
    }
  }
  hadamard_transform_lanes<Layout, COLUMNS, Layout::SECOND_LANE_STRIDE_LIMIT>(column_values, lane);
  METAL_PRAGMA_UNROLL
  for (uint value_stride = 1; value_stride < Layout::IN_THREAD_VALUES; value_stride <<= 1) {
    METAL_PRAGMA_UNROLL
    for (uint column = 0; column < COLUMNS; ++column) {
      METAL_PRAGMA_UNROLL
      for (uint block = 0; block < Layout::VALUES_PER_THREAD; block += 2 * value_stride) {
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

// TODO: a columns agnostic kernel
template <uint DIMENSION>
VARIANTS(DIMENSION, 4096, 5120, 6144, 6656, 17408, 19968)
PUBLIC KERNEL(TrellisTransform)(
    device const bfloat* input,
    device const float* rht_factors,
    device const float* mixing,
    device int8_t* activations,
    device float4* column_group_sums,
    device float* scales,
    constant uint& batch,
    threadgroup float scratch[TransformLayout<DIMENSION>::SCRATCH_SIZE],
    threadgroup float simdgroup_maxima[SIMDGROUPS_PER_THREADGROUP],
    threadgroup float shared_mixing[TransformLayout<DIMENSION>::MIXING_SIZE],
    const uint token GROUPS(batch),
    const uint thread_index THREADS(THREADS_PER_THREADGROUP),
    const ThreadContext thread_context
) {
  using Layout = TransformLayout<DIMENSION>;
  static_assert(
      Layout::HADAMARD_SIZE >= THREADS_PER_THREADGROUP && Layout::HADAMARD_SIZE <= 4096,
      "unsupported Hadamard size"
  );
  static_assert(Layout::THREADGROUP_BYTES <= THREADGROUP_MEMORY_LIMIT, "does not fit threadgroup memory");

  const ushort lane = ushort(thread_context.simd_lane_id);
  const ushort simdgroup = ushort(thread_context.simdgroup_index);

  for (uint index = thread_index; index < Layout::MIXING_DIMENSION * Layout::MIXING_DIMENSION;
       index += THREADS_PER_THREADGROUP) {
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

  const float maximum = reduce_activation_quantization_row_maximum<SIMDGROUPS_PER_THREADGROUP>(
      local_maximum,
      simdgroup_maxima,
      thread_context
  );
  const float scale = (isfinite(maximum) && maximum > 0.0f) ? precise::divide(maximum, 127.0f) : 1.0f;

  int local_column_group_sums[trellis::COLUMN_GROUP_COUNT] = {};
  METAL_PRAGMA_UNROLL
  for (uint mixing_index = 0; mixing_index < Layout::MIXING_DIMENSION; ++mixing_index) {
    METAL_PRAGMA_UNROLL
    for (uint value_index = 0; value_index < Layout::VALUES_PER_THREAD; ++value_index) {
      const uint hadamard_index = hadamard_index_after_transpose<Layout>(lane, simdgroup, value_index);
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
