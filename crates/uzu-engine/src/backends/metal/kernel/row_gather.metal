#include <metal_stdlib>
#include "common/defines.h"
#include "common/dsl.h"

using namespace metal;

PUBLIC KERNEL(RowGather)(
    device uint8_t* dst,
    const device uint8_t* src,
    // [len][indices...], deepest accepted row at word 1
    const device uint* accepted_indices,
    constant const uint& row_bytes,
    const uint byte_idx AXIS(row_bytes, 256)
) {
  const uint index = accepted_indices[1];
  dst[byte_idx] = src[index * row_bytes + byte_idx];
}
