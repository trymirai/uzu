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
void applyRope(
    device const T* qkv, // [suffix_len, (num_heads + 2*num_groups) * head_dim]
    device const T* cosines,          // [max_seq_len, head_dim]
    device const T* sines,            // [max_seq_len, head_dim]
    device const int* tokenPositions, // [suffix_len] - actual token positions
    device T* rotatedQueries,         // [num_heads,   suffix_len,  head_dim]
    device T* rotatedKeys,            // [num_groups,  suffix_len,  head_dim]
    constant int& headDim,
    constant int& numHeads,
    constant int& numGroups,
    constant int& suffixLength,
    constant int& maxSequenceLength,
    const uint3 position // x: headIdx, y: tokenIdx, z: dimIdx
) {
  const uint headIndex = position.x;
  const uint tokenIndex = position.y;
  const uint dimensionIndex = position.z;

  if (headIndex >= numHeads || tokenIndex >= suffixLength ||
      dimensionIndex >= headDim)
    return;
  if (headDim & 1)
    return; // head_dim must be even
  if (numHeads % numGroups != 0)
    return;

  const uint groupIndex =
      headIndex / (numHeads / numGroups); // which KV group this head belongs to
  const uint totalHeads = numHeads + 2 * numGroups;

  // Use actual token position from buffer
  const uint rawPosition = tokenPositions[tokenIndex];
  const uint absolutePosition =
      rawPosition > maxSequenceLength ? 0 : rawPosition;

  const uint halfDimension = headDim / 2;

  TensorView3D<const T> qkvTensorView =
      TensorView3D<const T>(qkv).shaped(suffixLength, totalHeads, headDim);
  TensorView2D<const T> cosinesTensorView =
      TensorView2D<const T>(cosines).shaped(maxSequenceLength, headDim);
  TensorView2D<const T> sinesTensorView =
      TensorView2D<const T>(sines).shaped(maxSequenceLength, headDim);
  TensorView3D<T> rotatedQueriesTensorView =
      TensorView3D<T>(rotatedQueries).shaped(numHeads, suffixLength, headDim);
  TensorView3D<T> rotatedKeysTensorView =
      TensorView3D<T>(rotatedKeys).shaped(numGroups, suffixLength, headDim);

  const T cosVal = cosinesTensorView(absolutePosition, dimensionIndex);
  const T sinVal = sinesTensorView(absolutePosition, dimensionIndex);

  /* ---------- QUERIES ---------- */
  T queryResult = applyRopeTransform(
      qkvTensorView,
      tokenIndex,
      headIndex,
      dimensionIndex,
      halfDimension,
      cosVal,
      sinVal
  );
  rotatedQueriesTensorView(headIndex, tokenIndex, dimensionIndex) = queryResult;

  /* ---------- KEYS & VALUES (only first head of each group processes)
   * ---------- */
  uint firstHeadInGroup = groupIndex * (numHeads / numGroups);
  if (headIndex == firstHeadInGroup) {
    /* ---------- keys ---------- */
    uint keyHeadIndex =
        numHeads + groupIndex; // Keys start after all query heads
    T keyResult = applyRopeTransform(
        qkvTensorView,
        tokenIndex,
        keyHeadIndex,
        dimensionIndex,
        halfDimension,
        cosVal,
        sinVal
    );
    rotatedKeysTensorView(groupIndex, tokenIndex, dimensionIndex) = keyResult;
  }
}

/* ---------- boiler-plate kernel wrappers ---------- */
#define outerArguments(T)                                                      \
  (device const T* qkv [[buffer(0)]],                                          \
   device const T* cosines [[buffer(1)]],                                      \
   device const T* sines [[buffer(2)]],                                        \
   device const int* tokenPositions [[buffer(3)]],                             \
   device T* rotatedQueries [[buffer(4)]],                                     \
   device T* rotatedKeys [[buffer(5)]],                                        \
   constant int& headDim [[buffer(6)]],                                        \
   constant int& numHeads [[buffer(7)]],                                       \
   constant int& numGroups [[buffer(8)]],                                      \
   constant int& suffixLength [[buffer(9)]],                                   \
   constant int& maxSequenceLength [[buffer(10)]],                             \
   const uint3 position [[thread_position_in_grid]])

#define innerArguments                                                         \
  (qkv,                                                                        \
   cosines,                                                                    \
   sines,                                                                      \
   tokenPositions,                                                             \
   rotatedQueries,                                                             \
   rotatedKeys,                                                                \
   headDim,                                                                    \
   numHeads,                                                                   \
   numGroups,                                                                  \
   suffixLength,                                                               \
   maxSequenceLength,                                                          \
   position)

generateKernels(32, applyRope)

#undef outerArguments
#undef innerArguments
