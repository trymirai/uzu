#include <metal_stdlib>
#include "common/defines.h"
#include "common/dsl.h"

using namespace metal;

PUBLIC KERNEL(AcceptedRowGather)(
    device uint8_t* dst,
    const device uint8_t* src,
    const device uint* accepted_indices,
    constant const uint& max_tokens,
    constant const uint& dst_stride,
    constant const uint& row_bytes,
    const uint elem_idx AXIS(max_tokens * row_bytes, 256)
) {
  const uint token = elem_idx / row_bytes;
  const uint offset = elem_idx % row_bytes;
  const uint accepted_len = accepted_indices[0];
  if (token >= accepted_len) {
    return;
  }
  const uint src_row = accepted_indices[accepted_len - token];
  dst[token * dst_stride + offset] = src[src_row * row_bytes + offset];
}
