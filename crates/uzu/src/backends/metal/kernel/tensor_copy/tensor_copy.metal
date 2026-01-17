#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
void tensorCopy(
    const device T* sourceBuffer,
    device T* destinationBuffer,
    constant int& length,
    const uint position
) {
  if (position < length) {
    destinationBuffer[position] = sourceBuffer[position];
  }
}

#define outerArguments(T)                                                      \
  (const device T* sourceBuffer [[buffer(0)]],                                 \
   device T* destinationBuffer [[buffer(1)]],                                  \
   constant int& length [[buffer(2)]],                                         \
   const uint position [[thread_position_in_grid]])

#define innerArguments (sourceBuffer, destinationBuffer, length, position)

generateKernels(32, tensorCopy)

#undef outerArguments
#undef innerArguments