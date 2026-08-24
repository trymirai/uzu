// Auto-generated from gpu_types/gated_act_mul - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::gated_act_mul {
enum class GatedActMulOp : uint32_t {
  FullPrecision = 0,
  Quantize = 1,
  QuantizeWithGroupSums = 2,
};
} // namespace uzu::gated_act_mul
