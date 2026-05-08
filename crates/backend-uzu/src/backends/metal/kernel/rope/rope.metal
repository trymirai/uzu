#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/tensor_view.h"

template <typename T>
inline T apply_rope_transform(
    TensorView3D<const T> qkv_tensor_view,
    uint token_index,
    uint head_index,
    uint dimension_index,
    uint rotary_pair_stride,
    T cos_val,
    T sin_val
) {
  T input_value = qkv_tensor_view(token_index, head_index, dimension_index);
  T paired_value = (dimension_index < rotary_pair_stride)
                       ? -qkv_tensor_view(
                             token_index,
                             head_index,
                             dimension_index + rotary_pair_stride
                         )
                       : qkv_tensor_view(
                             token_index,
                             head_index,
                             dimension_index - rotary_pair_stride
                         );

  return input_value * cos_val + paired_value * sin_val;
}

inline bool get_rotary_dimension_index(
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
    device const float* inverse_frequencies,
    device T* rotated_queries,          // [num_heads,   suffix_len,  head_dim]
    device T* rotated_keys,             // [num_groups,  suffix_len,  head_dim]
    constant uint& head_dim,
    constant uint& rope_dim,
    constant uint& rotary_pair_stride,
    constant uint& inverse_frequency_count,
    constant uint& rope_max_sequence_length,
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
  if (rope_dim != 0 && inverse_frequency_count < rope_dim / 2)
    return;
  if (rotary_pair_stride < rope_dim / 2)
    return;
  if (rotary_pair_stride + rope_dim / 2 > head_dim)
    return;
  if (num_groups == 0)
    return;
  if (num_heads % num_groups != 0)
    return;

  const uint group_index = head_index / (num_heads / num_groups);
  const uint total_heads = num_heads + 2 * num_groups;
  const uint raw_position = token_positions[token_index];
  const uint half_rope_dim = rope_dim / 2;

  const uint absolute_position =
      raw_position >= rope_max_sequence_length ? 0 : raw_position;

  TensorView3D<const T> qkv_tensor_view =
      TensorView3D<const T>(qkv).shaped(suffix_length, total_heads, head_dim);

  TensorView3D<T> rotated_queries_tensor_view =
      TensorView3D<T>(rotated_queries)
          .shaped(num_heads, suffix_length, head_dim);

  TensorView3D<T> rotated_keys_tensor_view =
      TensorView3D<T>(rotated_keys).shaped(num_groups, suffix_length, head_dim);

  uint first_head_in_group = group_index * (num_heads / num_groups);
  uint rotary_dimension_index = 0;

  if (get_rotary_dimension_index(
          dimension_index,
          half_rope_dim,
          rotary_pair_stride,
          rotary_dimension_index
      )) {
    const uint frequency_index = rotary_dimension_index % half_rope_dim;
    const float frequency = inverse_frequencies[frequency_index];
    const float angle = static_cast<float>(absolute_position) * frequency;

    const T cos_val =
        static_cast<T>(metal::fast::cos(angle) * rope_attention_scaling_factor);

    const T sin_val =
        static_cast<T>(metal::fast::sin(angle) * rope_attention_scaling_factor);

    T query_result = apply_rope_transform(
        qkv_tensor_view,
        token_index,
        head_index,
        dimension_index,
        rotary_pair_stride,
        cos_val,
        sin_val
    );

    rotated_queries_tensor_view(head_index, token_index, dimension_index) =
        query_result;

    if (head_index == first_head_in_group) {
      uint key_head_index = num_heads + group_index;
      T key_result = apply_rope_transform(
          qkv_tensor_view,
          token_index,
          key_head_index,
          dimension_index,
          rotary_pair_stride,
          cos_val,
          sin_val
      );

      rotated_keys_tensor_view(group_index, token_index, dimension_index) =
          key_result;
    }
  } else {
    rotated_queries_tensor_view(head_index, token_index, dimension_index) =
        qkv_tensor_view(token_index, head_index, dimension_index);

    if (head_index == first_head_in_group) {
      uint key_head_index = num_heads + group_index;
      rotated_keys_tensor_view(group_index, token_index, dimension_index) =
          qkv_tensor_view(token_index, key_head_index, dimension_index);
    }
  }
}
