#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
void swap(
    device T* buffer,
    const int sourceIdx,
    const int destIdx
) {
    const T temp = buffer[sourceIdx];
    buffer[sourceIdx] = buffer[destIdx];
    buffer[destIdx] = temp;
}

template <typename T>
void updateKVCache(
    device T* inPlaceKeys,
    device T* inPlaceValues,
    device int2* swaps,
    constant int& swapCount,
    constant int& numHeads,
    constant int& maxSequenceLength,
    constant int& headDim,
    const uint2 position
) {
    for (int i = 0; i < swapCount; ++i) {
        // [headIdx: 0..numHeads, tokenIdx: 0..maxSequenceLength, channelIdx: 0..headDim]
        const int headOffset = position.x * maxSequenceLength * headDim;
        const int channelOffset = position.y;
        const int sourceIdx = headOffset + swaps[i].x * headDim + channelOffset;
        const int destIdx = headOffset + swaps[i].y * headDim + channelOffset;

        swap(inPlaceKeys, sourceIdx, destIdx);
        swap(inPlaceValues, sourceIdx, destIdx);
    }
}

#define outerArguments(T)                                        \
(device T* inPlaceKeys [[ buffer(0) ]],                          \
device T* inPlaceValues [[ buffer(1) ]],                         \
device int2* swaps [[ buffer(2) ]],                              \
constant int& swapCount [[ buffer(3) ]],                         \
constant int& numHeads [[ buffer(4) ]],                          \
constant int& maxSequenceLength [[ buffer(5) ]],                 \
constant int& headDim [[ buffer(6) ]],                           \
const uint2 position [[ thread_position_in_grid ]])              \

#define innerArguments \
(inPlaceKeys,          \
inPlaceValues,         \
swaps,                 \
swapCount,             \
numHeads,              \
maxSequenceLength,     \
headDim,               \
position)              \

generateKernels(updateKVCache)

#undef outerArguments
#undef innerArguments

// Async scatter kernel: copies KV from suffix slot to ring buffer
// dest = (initial_ring_offset + pass_idx) % window_length
// source = window_length (suffix slot - always same position)
template <typename T>
void asyncScatterKV(
    device T* keys,
    device T* values,
    constant int& windowLength,
    constant int& initialRingOffset,
    constant int& passIdx,
    constant int& numHeads,
    constant int& headDim,
    const uint2 position
) {
    const int source = windowLength;  // suffix slot
    const int dest = (initialRingOffset + passIdx) % windowLength;

    const int maxSeqLen = windowLength + 1;  // ring + suffix
    const int headOffset = position.x * maxSeqLen * headDim;
    const int channelOffset = position.y;
    const int sourceIdx = headOffset + source * headDim + channelOffset;
    const int destIdx = headOffset + dest * headDim + channelOffset;

    keys[destIdx] = keys[sourceIdx];
    values[destIdx] = values[sourceIdx];
}

#define outerArguments(T)                                        \
(device T* keys [[ buffer(0) ]],                                 \
device T* values [[ buffer(1) ]],                                \
constant int& windowLength [[ buffer(2) ]],                      \
constant int& initialRingOffset [[ buffer(3) ]],                 \
constant int& passIdx [[ buffer(4) ]],                           \
constant int& numHeads [[ buffer(5) ]],                          \
constant int& headDim [[ buffer(6) ]],                           \
const uint2 position [[ thread_position_in_grid ]])              \

#define innerArguments \
(keys, values, windowLength, initialRingOffset, passIdx, numHeads, headDim, position)

generateKernels(asyncScatterKV)

#undef outerArguments
#undef innerArguments
