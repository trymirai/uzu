#pragma once

#include <metal_stdlib>
using namespace metal;

struct ThreadContext {
  uint simd_lane_id;    // [[thread_index_in_simdgroup]] — 0..simdgroup_size
  uint simdgroup_index; // [[simdgroup_index_in_threadgroup]] —
                        // 0..simdgroups_per_threadgroup
  uint simdgroup_size;  // [[threads_per_simdgroup]] — typically 32
  uint simdgroups_per_threadgroup; // [[simdgroups_per_threadgroup]]
  uint3 threadgroup_size;
  uint3 threadgroup_count;
  uint3 grid_size;
};
