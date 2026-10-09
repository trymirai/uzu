#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/packed.h"
#include "../generated/d4s4.h"
#include "../generated/embedding.h"
#include "../generated/quantization_method.h"
#include "../hadamard_transform/hadamard_transform.h"
#include "quantization.h"

using namespace metal;
using namespace uzu::embedding;
using namespace uzu::quantization_method;
using uzu::read_packed;

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(InputEmbeddingLookup)(
    const device uint* token_ids,
    const device uchar* values,
    const device T* scales OPTIONAL(table_kind != EmbeddingTableKind::Dense),
    const device uchar* zero_points OPTIONAL(table_kind == EmbeddingTableKind::Quantized && quantization_method == QuantizationMethod::ScaleZeroPoint),
    const device int* hadamard_factors OPTIONAL(use_hadamard),
    const device uchar* ladder_indices OPTIONAL(table_kind == EmbeddingTableKind::D4S4),
    const device half* ladder OPTIONAL(table_kind == EmbeddingTableKind::D4S4),
    const device char4* codebook OPTIONAL(table_kind == EmbeddingTableKind::D4S4),
    device T* output,
    constant uint& batch_size,
    constant uint& vocab_size,
    constant uint& model_dim,
    constant float& input_scale,
    const EmbeddingTableKind table_kind SPECIALIZE,
    const uint group_size SPECIALIZE_IF(table_kind == EmbeddingTableKind::Quantized),
    const uzu::quantization::QuantizationMode quantization_mode SPECIALIZE_IF(table_kind == EmbeddingTableKind::Quantized),
    const QuantizationMethod quantization_method SPECIALIZE_IF(table_kind == EmbeddingTableKind::Quantized),
    const bool use_hadamard SPECIALIZE,
    const uint dim_idx AXIS(model_dim, 256),
    const uint batch_idx AXIS(batch_size, 1)
) {
  const uint output_idx = batch_idx * model_dim + dim_idx;
  const uint token_id = token_ids[batch_idx];
  if (token_id >= vocab_size) {
    output[output_idx] = T(0);
    return;
  }

  float loaded;
  if (table_kind == EmbeddingTableKind::Dense) {
    loaded = float(reinterpret_cast<const device T*>(values)[token_id * model_dim + dim_idx] * T(input_scale));
  } else if (table_kind == EmbeddingTableKind::Quantized) {
    const bool is_u4 = quantization_mode == uzu::quantization::QuantizationMode::U4;
    const uint group_idx = dim_idx / group_size;
    const uint num_groups = (model_dim + group_size - 1) / group_size;
    const uint scale_idx = token_id * num_groups + group_idx;
    const float scale = float(scales[scale_idx]);
    const uint row = token_id * (is_u4 ? model_dim / 2 : model_dim);
    float code;
    if (is_u4) {
      code = float(read_packed<4>(values, 2 * row + dim_idx));
    } else if (quantization_mode == uzu::quantization::QuantizationMode::I8) {
      code = float(reinterpret_cast<const device char*>(values)[row + dim_idx]);
    } else {
      code = float(values[row + dim_idx]);
    }
    float bias;
    if (quantization_method == QuantizationMethod::ScaleZeroPoint) {
      const uint zero_point = is_u4 ? read_packed<4>(zero_points, 2 * token_id * ((num_groups + 1) / 2) + group_idx)
                                    : zero_points[scale_idx];
      bias = -scale * float(zero_point);
    } else {
      bias = -scale * (is_u4 ? 8.0f : 128.0f);
    }
    loaded = float(T((scale * code + bias) * input_scale));
  } else {
    const uint ladder_index = read_packed<4>(
        ladder_indices,
        token_id * (model_dim / uzu::d4s4::COLUMNS_PER_LADDER_SCALE) + dim_idx / uzu::d4s4::COLUMNS_PER_LADDER_SCALE
    );
    const char4 point =
        codebook[values[token_id * (model_dim / uzu::d4s4::VALUES_PER_CODE) + dim_idx / uzu::d4s4::VALUES_PER_CODE]];
    loaded = float(scales[token_id]) * float(ladder[ladder_index]) *
             float(point[dim_idx % uzu::d4s4::VALUES_PER_CODE]) * input_scale;
  }
  if (use_hadamard) {
    loaded = simdgroup_output_random_hadamard_transform(
        static_cast<ushort>(dim_idx % METAL_SIMD_SIZE),
        loaded,
        hadamard_factors[dim_idx]
    );
  }
  output[output_idx] = T(loaded);
}
