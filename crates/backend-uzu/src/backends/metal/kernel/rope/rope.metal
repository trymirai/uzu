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
    device const ElementT* qkv,         // [suffix_len, (num_heads + 2*num_groups) * head_dim]
    device const RopeT* cosines,        // [max_seq_len, rope_dim]
    device const RopeT* sines,          // [max_seq_len, rope_dim]
    device const int* token_positions,  // [suffix_len] - actual token positions
    device ElementT* rotated_queries,   // [num_heads,   suffix_len,  head_dim]
    device ElementT* rotated_keys,      // [num_groups,  suffix_len,  head_dim]
    constant uint& head_dim,
    constant uint& rope_dim,
    constant uint& num_heads,
    constant uint& num_groups,
    constant uint& suffix_length,
    constant uint& max_sequence_length,
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
      raw_position > max_sequence_length ? 0 : raw_position;
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
  TensorView3D<ElementT> rotated_keys_tensor_view =
      TensorView3D<ElementT>(rotated_keys)
          .shaped(num_groups, suffix_length, head_dim);

  uint first_head_in_group = group_index * (num_heads / num_groups);

  /* ---------- QUERIES ---------- */
  if (dimension_index < rope_dim) {
    const float cos_val =
        float(cosines_tensor_view(absolutePosition, dimension_index));
    const float sin_val =
        float(sines_tensor_view(absolutePosition, dimension_index));

    ElementT queryResult = applyRopeTransform(
        qkv_tensor_view,
        token_index,
        head_index,
        dimension_index,
        half_rope_dim,
        cos_val,
        sin_val
    );
    rotated_queries_tensor_view(head_index, token_index, dimension_index) =
        queryResult;

    /* ---------- KEYS (only first head of each group processes) ---------- */
    if (head_index == first_head_in_group) {
      uint key_head_index =
          num_heads + group_index; // Keys start after all query heads
      ElementT keyResult = applyRopeTransform(
          qkv_tensor_view,
          token_index,
          key_head_index,
          dimension_index,
          half_rope_dim,
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
