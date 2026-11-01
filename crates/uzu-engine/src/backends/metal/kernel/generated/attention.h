// Auto-generated from gpu_types/attention - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::attention {
typedef struct {
  uint32_t q_strides[3];
  uint32_t k_strides[3];
  uint32_t v_strides[3];
  uint32_t o_strides[3];
  uint32_t gqa_factor;
  float scale;
  uint32_t q_len;
  uint32_t k_len;
  uint32_t q_off;
  uint32_t nk;
} AttnParams;
} // namespace uzu::attention
