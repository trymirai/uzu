#include <metal_stdlib>
#include "../common/dsl.h"

template <typename T, typename BiasT>
VARIANTS(T, float, half, bfloat)
VARIANTS(BiasT, float, half, bfloat)
PUBLIC KERNEL(TensorAddBias)(
    const device T* input OPTIONAL(!in_place),
    const device BiasT* bias,
    const device uint* bias_row_indices OPTIONAL(indexed),
    device T* output,
    constant uint& num_cols,
    constant uint& length,
    const uint position AXIS(length, 32),
    const bool in_place SPECIALIZE,
    const bool indexed SPECIALIZE
) {
  if (in_place) {
    input = output;
  }

  const uint row = position / num_cols;
  const uint column = position % num_cols;
  const ulong bias_position = indexed ? ulong(bias_row_indices[row]) * ulong(num_cols) + ulong(column) : column;
  output[position] = static_cast<T>(static_cast<float>(input[position]) + static_cast<float>(bias[bias_position]));
}
