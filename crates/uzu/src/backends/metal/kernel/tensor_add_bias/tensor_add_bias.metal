#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
void tensorAddBias(
    const device T* input,
    const device T* bias,
    device T* output,
    constant int& numCols,
    constant int& length,
    const uint position
) {
  if (position < length) {
    int col = position % numCols;
    output[position] = input[position] + bias[col];
  }
}

#define outerArguments(T)                                                      \
  (const device T* input [[buffer(0)]],                                        \
   const device T* bias [[buffer(1)]],                                         \
   device T* output [[buffer(2)]],                                             \
   constant int& numCols [[buffer(3)]],                                        \
   constant int& length [[buffer(4)]],                                         \
   const uint position [[thread_position_in_grid]])

#define innerArguments (input, bias, output, numCols, length, position)

generateKernels(32, tensorAddBias)

#undef outerArguments
#undef innerArguments
