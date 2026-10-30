// Auto-generated from gpu_types/confidence_head - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::confidence_head {
typedef struct {
  uint32_t candidates;
  uint32_t children;
  uint32_t tree_slot_count;
  uint32_t vocab_size;
  float logit_cap;
  float epsilon;
} ConfidenceHeadParams;
} // namespace uzu::confidence_head
