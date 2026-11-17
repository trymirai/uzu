// Auto-generated from gpu_types/hadamard_order - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::hadamard_order {
static constant constexpr uint32_t HADAMARD_TRANSFORM_BLOCK_SIZE = 32;

static constant constexpr uint32_t SUPPORTED_HADAMARD_BLOCK_SIZES[2] =
    {HADAMARD_TRANSFORM_BLOCK_SIZE, 128};
} // namespace uzu::hadamard_order
