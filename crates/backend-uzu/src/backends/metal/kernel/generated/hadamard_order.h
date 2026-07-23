// Auto-generated from gpu_types/hadamard_order - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::hadamard_order {
static constant constexpr size_t HADAMARD_TRANSFORM_BLOCK_SIZE = 32;

enum class HadamardTransformOrder : uint32_t {
  Input = 0,
  Output = 1,
};
} // namespace uzu::hadamard_order
