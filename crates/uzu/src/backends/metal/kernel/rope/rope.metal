#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
inline T applyRopeTransform(
    TensorView3D<const T> qkv_tensor_view,
    uint token_idx,
    uint head_idx,
    uint dim_idx,
    uint half_dim,
    T cos_val,
    T sin_val
) {
  T inputVal = qkv_tensor_view(token_idx, head_idx, dim_idx);
  T pairedVal = (dim_idx < half_dim)
                    ? -qkv_tensor_view(token_idx, head_idx, dim_idx + half_dim)
                    : qkv_tensor_view(token_idx, head_idx, dim_idx - half_dim);
  return inputVal * cos_val + pairedVal * sin_val;
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(Rope)(
    device const T* qkv,                // [suffix_len, (num_heads + 2*num_groups) * head_dim]
    device const T* cosines,            // [max_seq_len, head_dim]
    device const T* sines,              // [max_seq_len, head_dim]
    device const int* token_positions,  // [suffix_len] - actual token positions
    device T* rotated_queries,          // [num_heads,   suffix_len,  head_dim]
    device T* rotated_keys,             // [num_heads,   suffix_len,  head_dim]
    constant uint& head_dim,
    constant uint& num_heads,
    constant uint& num_groups,
    constant uint& suffix_length,
    constant uint& max_sequence_length,
    const uint head_index AXIS(num_heads, 1),
    const uint token_index AXIS(suffix_length, 1),
    const uint dimension_index AXIS(head_dim, 32)
) {
  if (head_index >= num_heads || token_index >= suffix_length ||
      dimension_index >= head_dim)
    return;
  if (head_dim & 1)
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

  const uint half_dimension = head_dim / 2;

  TensorView3D<const T> qkv_tensor_view =
      TensorView3D<const T>(qkv).shaped(suffix_length, total_heads, head_dim);
  TensorView2D<const T> cosines_tensor_view =
      TensorView2D<const T>(cosines).shaped(max_sequence_length, head_dim);
  TensorView2D<const T> sines_tensor_view =
      TensorView2D<const T>(sines).shaped(max_sequence_length, head_dim);
  TensorView3D<T> rotated_queries_tensor_view =
      TensorView3D<T>(rotated_queries)
          .shaped(num_heads, suffix_length, head_dim);
  TensorView3D<T> rotated_keys_tensor_view =
      TensorView3D<T>(rotated_keys).shaped(num_groups, suffix_length, head_dim);

  const T cos_val = cosines_tensor_view(absolutePosition, dimension_index);
  const T sin_val = sines_tensor_view(absolutePosition, dimension_index);

  /* ---------- QUERIES ---------- */
  T queryResult = applyRopeTransform(
      qkv_tensor_view,
      token_index,
      head_index,
      dimension_index,
      half_dimension,
      cos_val,
      sin_val
  );
  rotated_queries_tensor_view(head_index, token_index, dimension_index) =
      queryResult;

  /* ---------- KEYS & VALUES (only first head of each group processes)
   * ---------- */
  uint first_head_in_group = group_index * (num_heads / num_groups);
  if (head_index == first_head_in_group) {
    /* ---------- keys ---------- */
    uint key_head_index =
        num_heads + group_index; // Keys start after all query heads
    T keyResult = applyRopeTransform(
        qkv_tensor_view,
        token_index,
        key_head_index,
        dimension_index,
        half_dimension,
        cos_val,
        sin_val
    );
    rotated_keys_tensor_view(group_index, token_index, dimension_index) =
        keyResult;
  }
}