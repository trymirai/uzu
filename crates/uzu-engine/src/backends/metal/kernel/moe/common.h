#pragma once

#include <metal_stdlib>

#define MOE_ROUTER_NUM_SIMDGROUPS 8
#define MOE_ROUTER_MAX_PER_THREAD 2
#define MOE_GEMV_NUM_SIMDGROUPS 8
#define MOE_GEMV_OUTPUT_ROW_TILE (32)

inline bool is_better_candidate(float v_new, uint id_new, float v_cur, uint id_cur, uint num_experts) {
  if (id_new >= num_experts)
    return false;
  if (id_cur >= num_experts)
    return true;
  return (v_new > v_cur) || (v_new == v_cur && id_new < id_cur);
}
