// Auto-generated from gpu_types/attention.rs - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::attention {
typedef struct {
  int64_t q_strides[3];
  int64_t k_strides[3];
  int64_t v_strides[3];
  int64_t o_strides[3];
  int32_t gqa_factor;
  float scale;
  int32_t q_len;
  int32_t k_len;
  int32_t q_off;
  int32_t nq_aligned;
  int32_t q_rem;
  int32_t nk;
  int32_t nk_aligned;
  int32_t k_rem;
} AttnParams;
} // namespace uzu::attention
