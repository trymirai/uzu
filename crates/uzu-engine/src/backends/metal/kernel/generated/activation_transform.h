// Auto-generated from gpu_types/activation_transform - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::activation_transform {
enum class ActivationTransformOp : uint32_t {
  InputRht = 0,
  OutputRht = 1,
  Quantize = 2,
  QuantizeWithGroupSums = 3,
};
} // namespace uzu::activation_transform
