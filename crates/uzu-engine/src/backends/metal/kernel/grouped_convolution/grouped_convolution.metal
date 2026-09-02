#include <metal_stdlib>
#include "../common/dsl.h"

template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(GroupedConvolution)(
    const device T* input,
    const device T* coefficients,
    const device T* base_kernel,
    device T* output,
    const constant uint& sequence_length,
    const constant uint& model_dim,
    const constant uint& groups,
    const constant uint& group_size,
    const constant uint& kernel_size,
    const constant uint& stage,
    const uint token AXIS(sequence_length, 1),
    const uint channel AXIS(model_dim, 256)
) {
  const uint group = channel / group_size;
  const uint coefficient_stride = 2 * kernel_size * groups;
  float value = 0.0f;
  for (uint tap = 0; tap < kernel_size && tap <= token; ++tap) {
    const uint base_index = (stage * kernel_size + tap) * model_dim + channel;
    const uint coefficient_index = token * coefficient_stride + stage * kernel_size * groups + tap * groups + group;
    const uint input_index = (token - tap) * model_dim + channel;
    value += (float(base_kernel[base_index]) + float(coefficients[coefficient_index])) * float(input[input_index]);
  }
  output[token * model_dim + channel] = static_cast<T>(value);
}
