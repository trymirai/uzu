#pragma once

#include <metal_stdlib>
using namespace metal;

struct ThreadContext {
  uint simdgroup_index;
  uint threadgroup_index;
  uint simdgroup_size;
  uint simdgroups_per_threadgroup;
  uint3 threadgroup_size;
  uint3 threadgroup_count;
  uint3 grid_size;
};
