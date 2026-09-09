#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

template<typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(SeparableCausalConv)(
    device const T* input,
    device const T* coefficient_deltas,
    device const T* weights,
    device const T* bias OPTIONAL(has_bias),
    device T* output,
    constant uint& sequence_length,
    constant uint& coefficient_row_stride,
    const uint model_dim SPECIALIZE,
    const uint kernel_size SPECIALIZE,
    const uint group_size SPECIALIZE,
    const bool has_bias SPECIALIZE,
    uint output_idx AXIS(sequence_length * model_dim, 256)
){
  const uint token = output_idx / model_dim;
  const uint channel = output_idx % model_dim;
  const uint num_groups = model_dim / group_size;
  const uint group = channel / group_size;

  float output_value = 0.0f;
  if (has_bias) {
    output_value = static_cast<float>(bias[channel]);
  }

  const uint available_tokens = min(kernel_size, token + 1);
  for (uint tokens_back = 0; tokens_back < available_tokens; tokens_back++) {
    const uint input_token = token - tokens_back;
    const uint input_index = input_token * model_dim + channel;
    const uint weight_index = channel * kernel_size + (kernel_size - 1 - tokens_back);
    const uint coefficient_index = token * coefficient_row_stride + tokens_back * num_groups + group;

    output_value +=
        float(input[input_index]) * (float(weights[weight_index]) + float(coefficient_deltas[coefficient_index]));
  }

  output[output_idx] = static_cast<T>(output_value);
}