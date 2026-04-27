// Auto-generated from gpu_types/attention.rs - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::attention {
typedef struct {
  uint64_t q_strides[3];
  uint64_t k_strides[3];
  uint64_t v_strides[3];
  uint64_t o_strides[3];
  uint32_t gqa_factor;
  float scale;
  uint32_t q_len;
  uint32_t k_len;
  uint32_t q_off;
  uint32_t nq_aligned;
  uint32_t q_rem;
  uint32_t nk;
  uint32_t nk_aligned;
  uint32_t k_rem;
} AttnParams;
} // namespace uzu::attention
