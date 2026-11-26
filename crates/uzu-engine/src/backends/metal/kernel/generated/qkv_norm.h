// Auto-generated from gpu_types/qkv_norm - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::qkv_norm {
enum class QKVL2Op : uint32_t {
  None = 0,
  RmsNorm = 1,
  Clip = 2,
};

typedef struct {
  uint32_t op[3];
  uint32_t full_layer[3];
  float epsilon[3];
  float scale_offset[3];
  float clip_threshold[3];
} QKVNormParams;
} // namespace uzu::qkv_norm
