#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;

#define VECTOR_WIDTH 4u
#define TOKENS_PER_THREADGROUP 1u
#define CHANNEL_VECTORS_PER_THREADGROUP 128u

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
    uint token AXIS(sequence_length, TOKENS_PER_THREADGROUP),
    uint channel_block_index AXIS((model_dim + VECTOR_WIDTH - 1u) / VECTOR_WIDTH, CHANNEL_VECTORS_PER_THREADGROUP)
){
  using ValueVector = vec<T, VECTOR_WIDTH>;
  using AccumulatorVector = vec<float, VECTOR_WIDTH>;

  const uint channel = channel_block_index * VECTOR_WIDTH;
  const uint num_groups = model_dim / group_size;
  const uint available_tokens = min(kernel_size, token + 1u);
  const uint group = channel / group_size;
  AccumulatorVector output_value = 0.0f;
  if (has_bias) {
    output_value = AccumulatorVector(*reinterpret_cast<const device ValueVector*>(bias + channel));
  }

  METAL_PRAGMA_UNROLL
  for (uint tokens_back = 0; tokens_back < available_tokens; ++tokens_back) {
    const uint input_token = token - tokens_back;
    const uint input_index = input_token * model_dim + channel;
    const uint stored_weight = kernel_size - 1u - tokens_back;
    const uint coefficient_index = token * coefficient_row_stride + tokens_back * num_groups + group;
    const AccumulatorVector weight = AccumulatorVector(
        float(weights[(channel + 0u) * kernel_size + stored_weight]),
        float(weights[(channel + 1u) * kernel_size + stored_weight]),
        float(weights[(channel + 2u) * kernel_size + stored_weight]),
        float(weights[(channel + 3u) * kernel_size + stored_weight])
    );

    output_value += (weight + float(coefficient_deltas[coefficient_index])) *
                    AccumulatorVector(*reinterpret_cast<const device ValueVector*>(input + input_index));
  }

  *reinterpret_cast<device ValueVector*>(output + token * model_dim + channel) = ValueVector(output_value);
}
