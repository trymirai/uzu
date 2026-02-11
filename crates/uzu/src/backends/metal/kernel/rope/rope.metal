#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
inline T applyRopeTransform(
    TensorView3D<const T> qkvTensorView,
    uint tokenIdx,
    uint headIdx,
    uint dimIdx,
    uint halfDim,
    T cosVal,
    T sinVal
) {
  T inputVal = qkvTensorView(tokenIdx, headIdx, dimIdx);
  T pairedVal = (dimIdx < halfDim)
                    ? -qkvTensorView(tokenIdx, headIdx, dimIdx + halfDim)
                    : qkvTensorView(tokenIdx, headIdx, dimIdx - halfDim);
  return inputVal * cosVal + pairedVal * sinVal;
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
  if (head_index >= num_heads 
      || token_index >= suffix_length 
      || dimension_index >= head_dim
      || head_dim & 1 != 0  // head_dim must be even
      || num_heads % num_groups != 0
  ) {
    return;
  }

  const uint groupIndex =
      head_index / (num_heads / num_groups); // which KV group this head belongs to
  const uint totalHeads = num_heads + 2 * num_groups;

  // Use actual token position from buffer
  const uint rawPosition = token_positions[token_index];
  const uint absolutePosition =
      rawPosition > max_sequence_length ? 0 : rawPosition;

  const uint halfDimension = head_dim / 2;

  TensorView3D<const T> qkvTensorView =
      TensorView3D<const T>(qkv).shaped(suffix_length, totalHeads, head_dim);
  TensorView2D<const T> cosinesTensorView =
      TensorView2D<const T>(cosines).shaped(max_sequence_length, head_dim);
  TensorView2D<const T> sinesTensorView =
      TensorView2D<const T>(sines).shaped(max_sequence_length, head_dim);
  TensorView3D<T> rotatedQueriesTensorView =
      TensorView3D<T>(rotated_queries).shaped(num_heads, suffix_length, head_dim);
  TensorView3D<T> rotatedKeysTensorView =
      TensorView3D<T>(rotated_keys).shaped(num_groups, suffix_length, head_dim);

  const T cosVal = cosinesTensorView(absolutePosition, dimension_index);
  const T sinVal = sinesTensorView(absolutePosition, dimension_index);

  /* ---------- QUERIES ---------- */
  T queryResult = applyRopeTransform(
      qkvTensorView,
      token_index,
      head_index,
      dimension_index,
      halfDimension,
      cosVal,
      sinVal
  );
  rotatedQueriesTensorView(head_index, token_index, dimension_index) = queryResult;

  /* ---------- KEYS & VALUES (only first head of each group processes)
  * ---------- */
  uint firstHeadInGroup = groupIndex * (num_heads / num_groups);
  if (head_index == firstHeadInGroup) {
    /* ---------- keys ---------- */
    uint keyHeadIndex =
        num_heads + groupIndex; // Keys start after all query heads
    T keyResult = applyRopeTransform(
        qkvTensorView,
        token_index,
        keyHeadIndex,
        dimension_index,
        halfDimension,
        cosVal,
        sinVal
    );
    rotatedKeysTensorView(groupIndex, token_index, dimension_index) = keyResult;
  }
}