#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/tensor_view.h"

template <typename ElementT>
inline ElementT applyRopeTransform(
    TensorView3D<const ElementT> qkv_tensor_view,
    uint token_idx,
    uint head_idx,
    uint dim_idx,
    uint half_dim,
    float cos_val,
    float sin_val
) {
  float inputVal = float(qkv_tensor_view(token_idx, head_idx, dim_idx));
  float pairedVal =
      (dim_idx < half_dim)
          ? -float(qkv_tensor_view(token_idx, head_idx, dim_idx + half_dim))
          : float(qkv_tensor_view(token_idx, head_idx, dim_idx - half_dim));
  return static_cast<ElementT>(inputVal * cos_val + pairedVal * sin_val);
}

template <typename ElementT, typename RopeT>
VARIANTS(ElementT, float, half, bfloat)
VARIANTS(RopeT, float, half, bfloat)
PUBLIC KERNEL(Rope)(
    device const ElementT* qkv,         // [suffix_len, total_heads, head_dim]
    device const RopeT* cosines,        // [max_seq_len, rope_dim]
    device const RopeT* sines,          // [max_seq_len, rope_dim]
    device const int* token_positions,  // [suffix_len]
    device ElementT* rotated_queries,   // [num_heads,  suffix_len, head_dim]
    device ElementT* rotated_keys OPTIONAL(!query_only),
    constant uint& head_dim,
    constant uint& rope_dim,
    constant uint& num_heads,
    constant uint& num_groups OPTIONAL(!query_only),
    constant uint& suffix_length,
    constant uint& max_sequence_length,
    const bool query_only SPECIALIZE,
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
  if (!query_only && (num_groups == 0 || num_heads % num_groups != 0))
    return;

  const uint total_heads = query_only ? num_heads : num_heads + 2 * num_groups;
  const uint raw_position = token_positions[token_index];
  const uint absolute_position =
      raw_position >= max_sequence_length ? 0 : raw_position;
  const uint half_rope_dim = rope_dim / 2;

  TensorView3D<const ElementT> qkv_tensor_view =
      TensorView3D<const ElementT>(qkv)
          .shaped(suffix_length, total_heads, head_dim);
  TensorView2D<const RopeT> cosines_tensor_view =
      TensorView2D<const RopeT>(cosines).shaped(max_sequence_length, rope_dim);
  TensorView2D<const RopeT> sines_tensor_view =
      TensorView2D<const RopeT>(sines).shaped(max_sequence_length, rope_dim);
  TensorView3D<ElementT> rotated_queries_tensor_view =
      TensorView3D<ElementT>(rotated_queries)
          .shaped(num_heads, suffix_length, head_dim);

  if (dimension_index < rope_dim) {
    const float cos_val =
        float(cosines_tensor_view(absolute_position, dimension_index));
    const float sin_val =
        float(sines_tensor_view(absolute_position, dimension_index));
    rotated_queries_tensor_view(head_index, token_index, dimension_index) =
        applyRopeTransform(
            qkv_tensor_view,
            token_index,
            head_index,
            dimension_index,
            half_rope_dim,
            cos_val,
            sin_val
        );
  } else {
    rotated_queries_tensor_view(head_index, token_index, dimension_index) =
        qkv_tensor_view(token_index, head_index, dimension_index);
  }

  if (query_only)
    return;

  const uint heads_per_group = num_heads / num_groups;
  const uint group_index = head_index / heads_per_group;
  if (head_index != group_index * heads_per_group)
    return;

  TensorView3D<ElementT> rotated_keys_tensor_view =
      TensorView3D<ElementT>(rotated_keys)
          .shaped(num_groups, suffix_length, head_dim);
  const uint key_head_index = num_heads + group_index;
  if (dimension_index < rope_dim) {
    const float cos_val =
        float(cosines_tensor_view(absolute_position, dimension_index));
    const float sin_val =
        float(sines_tensor_view(absolute_position, dimension_index));
    rotated_keys_tensor_view(group_index, token_index, dimension_index) =
        applyRopeTransform(
            qkv_tensor_view,
            token_index,
            key_head_index,
            dimension_index,
            half_rope_dim,
            cos_val,
            sin_val
        );
  } else {
    rotated_keys_tensor_view(group_index, token_index, dimension_index) =
        qkv_tensor_view(token_index, key_head_index, dimension_index);
  }
}
