#pragma once

#include <metal_stdlib>
using namespace metal;

struct ThreadContext {
  uint simdgroup_index;
  uint threadgroup_index;
  uint simdgroup_size;
  uint threadgroup_size;
  uint simdgroups_per_threadgroup;
};
