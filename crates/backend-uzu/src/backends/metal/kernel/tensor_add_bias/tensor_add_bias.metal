#include <metal_stdlib>
#include "../common/dsl.h"

template <typename T, typename BiasT>
VARIANTS(T, float, half, bfloat)
VARIANTS(BiasT, float, half, bfloat)
PUBLIC KERNEL(TensorAddBias)(
    const device T* input OPTIONAL(!in_place),
    const device BiasT* bias,
    device T* output,
    constant uint& num_cols,
    constant uint& length,
    const uint position AXIS(length, 32),
    const bool in_place SPECIALIZE
) {
  if (in_place) {
    input = output;
  }

  uint col = position % num_cols;
  output[position] = static_cast<T>(static_cast<float>(input[position]) + static_cast<float>(bias[col]));
}
