// Auto-generated from gpu_types/rope - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::rope {
enum class RoPEKind : uint32_t {
  None = 0,
  Sequential = 1,
  ContiguousGrid = 2,
  InterleavedGrid = 3,
};

typedef struct {
  uint32_t first_cell;
  uint32_t grid_width;
  uint32_t grid_height;
  uint32_t height_pairs;
  uint32_t width_pairs;
  uint32_t merge_size;
} RoPEGridParams;
} // namespace uzu::rope
