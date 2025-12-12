#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
void tensorAddSwap(
    device T* skipBuffer,
    device T* mainBuffer,
    constant int& length,
    const uint position
) {
  if (position < length) {
    T skipValue = skipBuffer[position];
    T mainValue = mainBuffer[position];
    T result = mainValue + skipValue;

    mainBuffer[position] = result;
    skipBuffer[position] = result;
  }
}

#define outerArguments(T)                                                      \
  (device T * skipBuffer [[buffer(0)]],                                        \
   device T * mainBuffer [[buffer(1)]],                                        \
   constant int& length [[buffer(2)]],                                         \
   const uint position [[thread_position_in_grid]])

#define innerArguments (skipBuffer, mainBuffer, length, position)

generateKernels(tensorAddSwap)

#undef outerArguments
#undef innerArguments