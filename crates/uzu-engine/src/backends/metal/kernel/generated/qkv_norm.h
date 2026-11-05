// Auto-generated from gpu_types/qkv_norm - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::qkv_norm {
typedef struct {
  uint32_t enabled[3];
  uint32_t full_layer[3];
  float epsilon[3];
  float scale_offset[3];
} QKVNormParams;
} // namespace uzu::qkv_norm
