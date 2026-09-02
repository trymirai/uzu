#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

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
    const uint token AXIS(sequence_length, 1),
    const uint channel_vector AXIS((model_dim + 3) / 4, 128)
) {
  const uint channel = channel_vector * 4;
  const uint groups = model_dim / group_size;
  const uint coefficient_stride = 2 * kernel_size * groups;

  if (group_size % 4 == 0 && channel + 4 <= model_dim) {
    const uint group = channel / group_size;
    float4 value = 0.0f;
#pragma unroll
    for (uint tap = 0; tap < kernel_size && tap <= token; ++tap) {
      const uint base_index = tap * model_dim + channel;
      const uint coefficient_index = token * coefficient_stride + tap * groups + group;
      const uint input_index = (token - tap) * model_dim + channel;
      value += (float4(*reinterpret_cast<const device vec<T, 4>*>(base_kernel + base_index)) +
                float(coefficients[coefficient_index])) *
               float4(*reinterpret_cast<const device vec<T, 4>*>(input + input_index));
    }
    *reinterpret_cast<device vec<T, 4>*>(output + token * model_dim + channel) = vec<T, 4>(value);
    return;
  }

  const uint channel_end = channel + 4 < model_dim ? channel + 4 : model_dim;
  for (uint scalar_channel = channel; scalar_channel < channel_end; ++scalar_channel) {
    const uint group = scalar_channel / group_size;
    float value = 0.0f;
#pragma unroll
    for (uint tap = 0; tap < kernel_size && tap <= token; ++tap) {
      const uint base_index = tap * model_dim + scalar_channel;
      const uint coefficient_index = token * coefficient_stride + tap * groups + group;
      const uint input_index = (token - tap) * model_dim + scalar_channel;
      value += (float(base_kernel[base_index]) + float(coefficients[coefficient_index])) * float(input[input_index]);
    }
    output[token * model_dim + scalar_channel] = static_cast<T>(value);
  }
}
