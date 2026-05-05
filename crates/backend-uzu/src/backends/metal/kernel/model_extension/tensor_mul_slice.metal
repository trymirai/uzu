#include <metal_stdlib>
#include "../common/dsl.h"

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(TensorMulSlice)(
    device T* values,
    const device T* slice_source,
    constant uint& suffix_length,
    constant uint& total_slice_dim,
    constant uint& slice_dim,
    constant uint& slice_index,
    const uint dim AXIS(slice_dim, 256),
    const uint token AXIS(suffix_length, 1)
) {
  const uint position = token * slice_dim + dim;
  const uint slice_offset = token * total_slice_dim + slice_index * slice_dim + dim;
  values[position] *= slice_source[slice_offset];
}
