#include <metal_stdlib>
#include "../common/dsl.h"
#include "../generated/quantization_method.h"
#include "../hadamard_transform/hadamard_transform.h"
#include "quantization.h"

using namespace uzu::quantization_method;
using namespace uzu::quantization;

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(QuantizedEmbeddingLookup) (
    const device uint64_t* token_ids,   // [batch_size]
    const device uint8_t* weights,      // [vocab_size, model_dim/packing_divisor] packed
    const device T* scales,             // [vocab_size, num_groups]
    const device uint8_t* zero_points OPTIONAL(quantization_method == QuantizationMethod::ScaleZeroPoint),
    const device T* biases OPTIONAL(quantization_method == QuantizationMethod::ScaleBias),
    device T* output,                   // [batch_size, model_dim]
    const device int32_t* output_hadamard_factors OPTIONAL(use_hadamard),
    constant uint32_t& batch_size,
    constant uint32_t& vocab_size,
    constant uint32_t& model_dim,
    constant float& input_scale,
    const uint32_t group_size SPECIALIZE,
    const QuantizationMode quantization_mode SPECIALIZE,
    const QuantizationMethod quantization_method SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    const uint dim_idx AXIS(model_dim, 32),
    const uint batch_idx AXIS(batch_size, 1)
) {
  const uint thread_position_in_grid = batch_idx * model_dim + dim_idx;
  const uint64_t token_id = token_ids[batch_idx];
  if (token_id >= vocab_size) {
    output[thread_position_in_grid] = T(0);
    return;
  }

  const uint group_idx = dim_idx / group_size;
  const uint num_groups = (model_dim + group_size - 1) / group_size;
  const uint zp_stride = quantization_mode == QuantizationMode::U4 ? (num_groups + 1) / 2 : num_groups;

  const T scale = scales[token_id * num_groups + group_idx];

  const uint packing_divisor = quantization_mode == QuantizationMode::U4 ? 2 : 1;
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
    const device int8_t* weights_i8 = reinterpret_cast<const device int8_t*>(weights);
    quantized_value = int(weights_i8[elem_idx]);
    break;
  };
  case QuantizationMode::U8: {
    const uint elem_idx = token_id * weights_stride + dim_idx;
    quantized_value = int(weights[elem_idx]);
    break;
  };
  }

  float bias = 0.0f;
  if (quantization_method == QuantizationMethod::ScaleBias) {
    bias = float(biases[token_id * num_groups + group_idx]);
  } else if (quantization_method == QuantizationMethod::ScaleZeroPoint) {
    uint zero_point = 0;
    if (quantization_mode == QuantizationMode::U4) {
      const uint8_t packed = zero_points[token_id * zp_stride + group_idx / 2];
      zero_point = (group_idx & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    } else {
      zero_point = zero_points[token_id * zp_stride + group_idx];
    }
    bias = -float(scale) * float(zero_point);
  } else {
    const uint midpoint = quantization_mode == QuantizationMode::U4 ? 8 : 128;
    bias = -float(scale) * float(midpoint);
  }

  float out_f = float(scale) * float(quantized_value) + bias;
  out_f *= input_scale;
  T out = T(out_f);
  if (use_hadamard) {
    out = simdgroup_output_random_hadamard_transform(
        static_cast<ushort>(dim_idx % METAL_SIMD_SIZE),
        out,
        output_hadamard_factors[dim_idx]
    );
  }
  output[thread_position_in_grid] = out;
}
