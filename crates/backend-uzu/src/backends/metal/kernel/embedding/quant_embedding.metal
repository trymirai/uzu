#include <metal_stdlib>
#include "../common/dsl.h"
#include "quantization.h"

using namespace uzu::quantization;

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(QuantizedEmbeddingLookup) (
    const device uint64_t* token_ids,   // [batch_size]
    const device uint8_t* weights,      // [vocab_size, model_dim/packing_divisor] packed
    const device T* scales,             // [vocab_size, num_groups]
    const device T* biases,             // [vocab_size, num_groups]
    device T* output,                   // [batch_size, model_dim]
    constant uint32_t& batch_size,
    constant uint32_t& vocab_size,
    constant uint32_t& model_dim,
    constant float& input_scale,
    const uint32_t group_size SPECIALIZE,
    const uint32_t quant_mode SPECIALIZE,
    const uint dim_idx AXIS(model_dim, 16),
    const uint batch_idx AXIS(batch_size, 16)
) {
  const QuantizationMode quantization_mode = QuantizationMode(
      quant_mode
  ); // TODO: should be accepted as a kernel argument
  const uint thread_position_in_grid = batch_idx * model_dim + dim_idx;
  const uint64_t token_id = token_ids[batch_idx];
  if (token_id >= vocab_size) {
    output[thread_position_in_grid] = T(0);
    return;
  }

  const uint group_idx = dim_idx / group_size;
  const uint num_groups = (model_dim + group_size - 1) / group_size;

  const T scale = scales[token_id * num_groups + group_idx];
  const T bias = biases[token_id * num_groups + group_idx];

  const uint packing_divisor =
      quantization_mode == QuantizationMode::U4 ? 2 : 1;
  const uint weights_stride = model_dim / packing_divisor;

  int quantized_value = 0;
  switch (quantization_mode) {
  case QuantizationMode::U4: {
    const uint byte_idx = token_id * weights_stride + (dim_idx / 2);
    const uint8_t packed = weights[byte_idx];
    if ((dim_idx & 1) == 0) {
      quantized_value = int(packed & 0x0F);
    } else {
      quantized_value = int((packed >> 4) & 0x0F);
    }
    break;
  };
  case QuantizationMode::I8: {
    const uint elem_idx = token_id * weights_stride + dim_idx;
    const device int8_t* weights_i8 =
        reinterpret_cast<const device int8_t*>(weights);
    quantized_value = int(weights_i8[elem_idx]);
    break;
  };
  case QuantizationMode::U8: {
    const uint elem_idx = token_id * weights_stride + dim_idx;
    quantized_value = int(weights[elem_idx]);
    break;
  };
  }

  float out_f = float(scale) * float(quantized_value) + float(bias);
  out_f *= input_scale;
  output[thread_position_in_grid] = T(out_f);
}
