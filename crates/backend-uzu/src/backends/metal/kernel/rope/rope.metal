#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/tensor_view.h"

inline float linearRampFactor(float min_value, float max_value, float dim) {
  if (min_value == max_value) {
    max_value += 0.001;
  }
  return metal::clamp((dim - min_value) / (max_value - min_value), 0.0, 1.0);
}

inline float2 yarnCorrectionRange(
    float beta_fast,
    float beta_slow,
    uint dim,
    float base,
    uint original_context_length,
    uint truncate
) {
  const float two_pi = 6.28318530717958647692;
  const float scale = static_cast<float>(dim) / (2.0 * metal::log(base));
  float low = scale * metal::log(
                          static_cast<float>(original_context_length) /
                          (beta_fast * two_pi)
                      );
  float high = scale * metal::log(
                           static_cast<float>(original_context_length) /
                           (beta_slow * two_pi)
                       );

  if (truncate != 0) {
    low = metal::floor(low);
    high = metal::ceil(high);
  }

  return float2(metal::max(low, 0.0), metal::min(high, static_cast<float>(dim - 1)));
}

inline float inverseFrequency(
    uint frequency_index,
    uint rotary_frequency_dim,
    uint rope_scaling_type,
    float rope_base,
    float rope_scaling_factor,
    uint rope_original_context_length,
    float rope_low_frequency_factor,
    float rope_high_frequency_factor,
    float rope_beta_fast,
    float rope_beta_slow,
    uint rope_truncate
) {
  const float two_pi = 6.28318530717958647692;
  const float exponent =
      static_cast<float>(2 * frequency_index) /
      static_cast<float>(rotary_frequency_dim);
  float value = metal::exp2(-exponent * metal::log2(rope_base));

  if (rope_scaling_type == 1) {
    return value / rope_scaling_factor;
  }

  if (rope_scaling_type == 2) {
    const float low_frequency_wavelength =
        static_cast<float>(rope_original_context_length) /
        rope_low_frequency_factor;
    const float high_frequency_wavelength =
        static_cast<float>(rope_original_context_length) /
        rope_high_frequency_factor;
    const float wavelength = two_pi / value;
    const float scaled = value / rope_scaling_factor;

    if (wavelength > low_frequency_wavelength) {
      return scaled;
    }
    if (wavelength >= high_frequency_wavelength) {
      float smoothing =
          static_cast<float>(rope_original_context_length) / wavelength -
          rope_low_frequency_factor;
      smoothing /=
          rope_high_frequency_factor - rope_low_frequency_factor;
      return smoothing * value + (1.0 - smoothing) * scaled;
    }
    return value;
  }

  if (rope_scaling_type == 3) {
    const float scaled = value / rope_scaling_factor;
    const float2 correction_range = yarnCorrectionRange(
        rope_beta_fast,
        rope_beta_slow,
        rotary_frequency_dim,
        rope_base,
        rope_original_context_length,
        rope_truncate
    );
    const float ramp =
        linearRampFactor(correction_range.x, correction_range.y, static_cast<float>(frequency_index));
    const float smoothing = 1.0 - ramp;
    return scaled * (1.0 - smoothing) + value * smoothing;
  }

  return value;
}

template <typename T>
inline T applyRopeTransform(
    TensorView3D<const T> qkv_tensor_view,
    uint token_idx,
    uint head_idx,
    uint dim_idx,
    uint rotary_pair_stride,
    T cos_val,
    T sin_val
) {
  T inputVal = qkv_tensor_view(token_idx, head_idx, dim_idx);
  T pairedVal =
      (dim_idx < rotary_pair_stride)
          ? -qkv_tensor_view(token_idx, head_idx, dim_idx + rotary_pair_stride)
          : qkv_tensor_view(token_idx, head_idx, dim_idx - rotary_pair_stride);
  return inputVal * cos_val + pairedVal * sin_val;
}

inline bool getRotaryDimensionIndex(
    uint dimension_index,
    uint half_rope_dim,
    uint rotary_pair_stride,
    thread uint& rotary_dimension_index
) {
  if (dimension_index < half_rope_dim) {
    rotary_dimension_index = dimension_index;
    return true;
  }

  if (dimension_index >= rotary_pair_stride &&
      dimension_index < rotary_pair_stride + half_rope_dim) {
    rotary_dimension_index =
        dimension_index - rotary_pair_stride + half_rope_dim;
    return true;
  }

  return false;
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(Rope)(
    device const T* qkv,                // [suffix_len, (num_heads + 2*num_groups) * head_dim]
    device const int* token_positions,  // [suffix_len] - actual token positions
    device T* rotated_queries,          // [num_heads,   suffix_len,  head_dim]
    device T* rotated_keys,             // [num_groups,  suffix_len,  head_dim]
    constant uint& head_dim,
    constant uint& rope_dim,
    constant uint& rotary_pair_stride,
    constant uint& rotary_frequency_dim,
    constant uint& rope_max_sequence_length,
    constant uint& rope_scaling_type,
    constant float& rope_base,
    constant float& rope_scaling_factor,
    constant uint& rope_original_context_length,
    constant float& rope_low_frequency_factor,
    constant float& rope_high_frequency_factor,
    constant float& rope_beta_fast,
    constant float& rope_beta_slow,
    constant uint& rope_truncate,
    constant float& rope_attention_scaling_factor,
    constant uint& num_heads,
    constant uint& num_groups,
    constant uint& suffix_length,
    const uint head_index AXIS(num_heads, 1),
    const uint token_index AXIS(suffix_length, 1),
    const uint dimension_index AXIS(head_dim, 128)
) {
  if (head_index >= num_heads || token_index >= suffix_length ||
      dimension_index >= head_dim)
    return;
  if (head_dim & 1)
    return;
  if (rope_dim & 1)
    return;
  if (rope_dim > head_dim)
    return;
  if (rotary_frequency_dim == 0)
    return;
  if (rotary_pair_stride < rope_dim / 2)
    return;
  if (rotary_pair_stride + rope_dim / 2 > head_dim)
    return;
  if (num_groups == 0)
    return;
  if (num_heads % num_groups != 0)
    return;

  const uint group_index =
      head_index /
      (num_heads / num_groups); // which KV group this head belongs to
  const uint total_heads = num_heads + 2 * num_groups;
  // Use actual token position from buffer
  const uint raw_position = token_positions[token_index];
  const uint absolutePosition =
      raw_position >= rope_max_sequence_length ? 0 : raw_position;
  const uint half_rope_dim = rope_dim / 2;

  TensorView3D<const T> qkv_tensor_view =
      TensorView3D<const T>(qkv).shaped(suffix_length, total_heads, head_dim);
  TensorView3D<T> rotated_queries_tensor_view =
      TensorView3D<T>(rotated_queries)
          .shaped(num_heads, suffix_length, head_dim);
  TensorView3D<T> rotated_keys_tensor_view =
      TensorView3D<T>(rotated_keys).shaped(num_groups, suffix_length, head_dim);

  uint first_head_in_group = group_index * (num_heads / num_groups);

  /* ---------- QUERIES ---------- */
  uint rotary_dimension_index = 0;
  if (getRotaryDimensionIndex(
          dimension_index,
          half_rope_dim,
          rotary_pair_stride,
          rotary_dimension_index
      )) {
    const uint frequency_index = rotary_dimension_index % half_rope_dim;
    const float frequency = inverseFrequency(
        frequency_index,
        rotary_frequency_dim,
        rope_scaling_type,
        rope_base,
        rope_scaling_factor,
        rope_original_context_length,
        rope_low_frequency_factor,
        rope_high_frequency_factor,
        rope_beta_fast,
        rope_beta_slow,
        rope_truncate
    );
    const float angle = static_cast<float>(absolutePosition) * frequency;
    const T cos_val =
        static_cast<T>(metal::fast::cos(angle) * rope_attention_scaling_factor);
    const T sin_val =
        static_cast<T>(metal::fast::sin(angle) * rope_attention_scaling_factor);

    T queryResult = applyRopeTransform(
        qkv_tensor_view,
        token_index,
        head_index,
        dimension_index,
        rotary_pair_stride,
        cos_val,
        sin_val
    );
    rotated_queries_tensor_view(head_index, token_index, dimension_index) =
        queryResult;

    /* ---------- KEYS (only first head of each group processes) ---------- */
    if (head_index == first_head_in_group) {
      uint key_head_index =
          num_heads + group_index; // Keys start after all query heads
      T keyResult = applyRopeTransform(
          qkv_tensor_view,
          token_index,
          key_head_index,
          dimension_index,
          rotary_pair_stride,
          cos_val,
          sin_val
      );
      rotated_keys_tensor_view(group_index, token_index, dimension_index) =
          keyResult;
    }
  } else {
    rotated_queries_tensor_view(head_index, token_index, dimension_index) =
        qkv_tensor_view(token_index, head_index, dimension_index);

    if (head_index == first_head_in_group) {
      uint key_head_index =
          num_heads + group_index; // Keys start after all query heads
      rotated_keys_tensor_view(group_index, token_index, dimension_index) =
          qkv_tensor_view(token_index, key_head_index, dimension_index);
    }
  }
}
