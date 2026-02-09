#include <metal_stdlib>
#include "../definitions.metal"

enum QuantizationMode : uint {
  QUANT_UINT4 = 0,
  QUANT_INT8 = 1,
  QUANT_UINT8 = 2
};

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(QuantizedEmbeddingLookup) (
    const device uint64_t* token_ids,   // [batch_size]
    const device uint8_t* weights,      // [vocab_size, model_dim/packing_divisor] packed
    const device T* scales,             // [vocab_size, num_groups]
    const device T* biases,             // [vocab_size, num_groups]
    device T* output,                   // [batch_size, model_dim]
    constant uint32_t& batch_size,
    constant uint32_t& vocab_size,
    constant uint32_t& model_dim,
    constant uint32_t& group_size,
    constant float& input_scale,
    const constant uint& quant_mode,
    const uint dim_idx AXIS(model_dim, 16),
    const uint batch_idx AXIS(batch_size, 16)
) {
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

  const uint packing_divisor = quant_mode == QUANT_UINT4 ? 2 : 1;
  const uint weights_stride = model_dim / packing_divisor;

  int quantized_value = 0;
  if (quant_mode == QUANT_UINT4) {
    const uint byte_idx = token_id * weights_stride + (dim_idx / 2);
    const uint8_t packed = weights[byte_idx];
    if ((dim_idx & 1) == 0) {
      quantized_value = int(packed & 0x0F);
    } else {
      quantized_value = int((packed >> 4) & 0x0F);
    }
  } else if (quant_mode == QUANT_INT8) {
    const uint elem_idx = token_id * weights_stride + dim_idx;
    const device int8_t* weights_i8 = reinterpret_cast<const device int8_t*>(weights);
    quantized_value = int(weights_i8[elem_idx]);
  } else {
    const uint elem_idx = token_id * weights_stride + dim_idx;
    quantized_value = int(weights[elem_idx]);
  }

  float out_f = float(scale) * float(quantized_value) + float(bias);
  out_f *= input_scale;
  output[thread_position_in_grid] = T(out_f);
}