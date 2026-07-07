#pragma once

#include "../../common/defines.h"
#include <metal_stdlib>

using namespace metal;

METAL_FUNC uint gdn_groups_per_key_head(uint num_v_heads, uint num_k_heads) { return num_v_heads / num_k_heads; }

METAL_FUNC uint gdn_key_head_for_value_head(uint hv_idx, uint num_v_heads, uint num_k_heads) {
  return hv_idx / gdn_groups_per_key_head(num_v_heads, num_k_heads);
}
