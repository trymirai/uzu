#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/grouped_convolution.h"

using namespace metal;

#define VECTOR_WIDTH 4u
#define TOKENS_PER_THREADGROUP 1u
#define CHANNEL_VECTORS_PER_THREADGROUP 128u

template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(GroupedConvolution)(
    const device T* input,
    const device T* coefficients,
    const device T* base_kernel,
    device T* output,
    const constant uint& sequence_length,
    const uint model_dim SPECIALIZE,
    const uint group_size SPECIALIZE,
    const uint kernel_size SPECIALIZE,
    const uint token AXIS(sequence_length, TOKENS_PER_THREADGROUP),
    const uint channel_vector AXIS((model_dim + VECTOR_WIDTH - 1u) / VECTOR_WIDTH, CHANNEL_VECTORS_PER_THREADGROUP)
) {
  using ValueVector = vec<T, VECTOR_WIDTH>;
  using AccumulatorVector = vec<float, VECTOR_WIDTH>;

  const uint channel = channel_vector * VECTOR_WIDTH;
  const uint groups = model_dim / group_size;
  const uint coefficient_stride = uzu::grouped_convolution::CONVOLUTION_STAGE_COUNT * kernel_size * groups;
  const uint tap_count = min(kernel_size, token + 1u);

  if (model_dim % VECTOR_WIDTH == 0 && group_size % VECTOR_WIDTH == 0) {
    const uint group = channel / group_size;
    AccumulatorVector value = 0.0f;
    METAL_PRAGMA_UNROLL
    for (uint tap = 0; tap < tap_count; ++tap) {
      const uint base_index = tap * model_dim + channel;
      const uint coefficient_index = token * coefficient_stride + tap * groups + group;
      const uint input_index = (token - tap) * model_dim + channel;
      value += (AccumulatorVector(*reinterpret_cast<const device ValueVector*>(base_kernel + base_index)) +
                float(coefficients[coefficient_index])) *
               AccumulatorVector(*reinterpret_cast<const device ValueVector*>(input + input_index));
    }
    *reinterpret_cast<device ValueVector*>(output + token * model_dim + channel) = ValueVector(value);
    return;
  }

  const uint channel_end = min(channel + VECTOR_WIDTH, model_dim);
  for (uint scalar_channel = channel; scalar_channel < channel_end; ++scalar_channel) {
    const uint group = scalar_channel / group_size;
    float value = 0.0f;
    METAL_PRAGMA_UNROLL
    for (uint tap = 0; tap < tap_count; ++tap) {
      const uint base_index = tap * model_dim + scalar_channel;
      const uint coefficient_index = token * coefficient_stride + tap * groups + group;
      const uint input_index = (token - tap) * model_dim + scalar_channel;
      value += (float(base_kernel[base_index]) + float(coefficients[coefficient_index])) * float(input[input_index]);
    }
    output[token * model_dim + scalar_channel] = static_cast<T>(value);
  }
}
