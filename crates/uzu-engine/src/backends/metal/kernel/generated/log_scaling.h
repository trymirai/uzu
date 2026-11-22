// Auto-generated from gpu_types/log_scaling - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::log_scaling {
typedef struct {
  float n_floor;
  float alpha;
  uint32_t position_offset;
} LogScalingParams;
} // namespace uzu::log_scaling
