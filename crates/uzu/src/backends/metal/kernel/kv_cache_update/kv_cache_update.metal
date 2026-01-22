#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
void swap(device T* buffer, const int sourceIdx, const int destIdx) {
  const T temp = buffer[sourceIdx];
  buffer[sourceIdx] = buffer[destIdx];
  buffer[destIdx] = temp;
}

template <typename T>
void updateKVCache(
    device T* inPlaceKeys,
    device T* inPlaceValues,
    const constant int2* swaps,
    const constant int& swapCount,
    const constant int& numHeads,
    const constant int& maxSequenceLength,
    const constant int& headDim,
    const uint2 position
) {
  for (int i = 0; i < swapCount; ++i) {
    // [headIdx: 0..numHeads, tokenIdx: 0..maxSequenceLength, channelIdx:
    // 0..headDim]
    const int headOffset = position.x * maxSequenceLength * headDim;
    const int channelOffset = position.y;
    const int sourceIdx = headOffset + swaps[i].x * headDim + channelOffset;
    const int destIdx = headOffset + swaps[i].y * headDim + channelOffset;

    swap(inPlaceKeys, sourceIdx, destIdx);
    swap(inPlaceValues, sourceIdx, destIdx);
  }
}

#define outerArguments(T)                                                      \
  (device T * inPlaceKeys [[buffer(0)]],                                       \
   device T * inPlaceValues [[buffer(1)]],                                     \
   const constant int2 * swaps [[buffer(2)]],                                  \
   const constant int& swapCount [[buffer(3)]],                                \
   const constant int& numHeads [[buffer(4)]],                                 \
   const constant int& maxSequenceLength [[buffer(5)]],                        \
   const constant int& headDim [[buffer(6)]],                                  \
   const uint2 position [[thread_position_in_grid]])

#define innerArguments                                                         \
  (inPlaceKeys,                                                                \
   inPlaceValues,                                                              \
   swaps,                                                                      \
   swapCount,                                                                  \
   numHeads,                                                                   \
   maxSequenceLength,                                                          \
   headDim,                                                                    \
   position)

generateKernels(1024, updateKVCache)

#undef outerArguments
#undef innerArguments
