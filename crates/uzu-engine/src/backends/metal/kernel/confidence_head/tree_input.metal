#include <metal_stdlib>
#include "../common/dsl.h"
#include "../weaver/weaver_frontier.h"

using namespace metal;

PUBLIC KERNEL(ConfidenceHeadTreeInput)(
    const device bfloat* depth_embedding,
    const device bfloat* child_in,
    const device bfloat* node_in,
    const device uint* frontier,
    const device uint* row_slots,
    device bfloat* output,
    const constant uint& rows,
    const constant uint& width,
    const uint index AXIS(rows * width, 256)
) {
  const uint row = index / width;
  const uint unit = index % width;
  const uint depth = row == 0 ? 0u : frontier[uint(FrontierIdx::Depth) * (rows - 1) + row - 1];
  float value = 0.0f;
  if (row == 0 || depth != 0) {
    value = float(depth_embedding[depth * width + unit]);
    if (row != 0) {
      value += float(child_in[(row - 1) * width + unit]);
    }
    const uint slot = row_slots[row];
    if (slot != 0xffffffffu) {
      value += float(node_in[slot * width + unit]);
    }
  }
  output[index] = bfloat(value);
}
