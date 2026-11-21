#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "patch_grid.h"

// `index`-th of `count` points spread evenly over [0, side - 1], computed as torch.linspace does:
// the first half steps up from 0, the second half steps down from side - 1
METAL_FUNC float vision_linspace(uint index, uint count, uint side) {
  if (count <= 1) {
    return 0.0f;
  }

  const float end = static_cast<float>(side - 1);
  const float step = end / static_cast<float>(count - 1);
  return index < count / 2 ? static_cast<float>(index) * step : end - static_cast<float>(count - index - 1) * step;
}

// Adds the learned position embedding, bilinearly resized from side x side to the patch grid, to the patch embeddings.
// The table holds embeddings for a side x side patch grid; other grids get them bilinearly resized,
// like resizing an image
//
// @param table        — learned position embeddings [side * side, model_dim]
// @param stream       — patch embeddings [rows, model_dim], updated in place
// @param side         — side of the learned position grid (48)
// @param grid_width   — patch grid width
// @param grid_height  — patch grid height
// @param merge_size   — block side size for merger, sets the row order (how many patches merge in 1 token)
// @param model_dim    — embedding size
// @param length       — number of elements in stream (rows * model_dim), one thread per 4 consecutive channels;
//                       model_dim must be a multiple of 4
template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(VisionPositionEmbedding)(
    const device T* table,
    device T* stream,
    constant uint& side,
    constant uint& grid_width,
    constant uint& grid_height,
    constant uint& merge_size,
    constant uint& model_dim,
    constant uint& length,
    const uint output_index AXIS(length / 4, 256)
) {
  const uint first_element = output_index * 4;
  const uint row = first_element / model_dim;
  const uint channel = first_element % model_dim;

  const uint2 patch_position = vision_patch_coordinates(row, grid_width, merge_size);
  const float y = vision_linspace(patch_position.y, grid_height, side);
  const float x = vision_linspace(patch_position.x, grid_width, side);
  const uint y_floor = static_cast<uint>(y);
  const uint x_floor = static_cast<uint>(x);
  const uint y_ceil = min(y_floor + 1, side - 1);
  const uint x_ceil = min(x_floor + 1, side - 1);

  // 2d coordinate transforms into row number from table
  const uint corners[4] = {
      y_floor * side + x_floor,
      y_floor * side + x_ceil,
      y_ceil * side + x_floor,
      y_ceil * side + x_ceil,
  };

  // bilinear weights: the closer a point is to a neighbor, the greater that neighbor's weight
  const float dy = y - static_cast<float>(y_floor);
  const float dx = x - static_cast<float>(x_floor);
  const float weights[4] = {
      (1.0f - dy) * (1.0f - dx),
      (1.0f - dy) * dx,
      dy * (1.0f - dx),
      dy * dx,
  };

  // accumulated in float, rounded to T once when added to the stream
  float4 embedding = float4(0.0f);
  METAL_PRAGMA_UNROLL
  for (uint i = 0; i < 4; ++i) {
    const uint table_idx = corners[i] * model_dim + channel;
    embedding += float4(*reinterpret_cast<const device vec<T, 4>*>(table + table_idx)) * weights[i];
  }
  device vec<T, 4>* output = reinterpret_cast<device vec<T, 4>*>(stream + first_element);
  *output = vec<T, 4>(float4(*output) + embedding);
}