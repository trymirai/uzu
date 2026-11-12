// Auto-generated from gpu_types/moe - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::moe {
static constant constexpr uint32_t ROUTER_MAX_TOP_K = 8;

static constant constexpr uint32_t ROUTER_MAX_EXPERTS = 256;

static constant constexpr uint32_t GEMV_INPUT_BLOCK_BYTES = 256;

static constant constexpr uint32_t COMBINE_BLOCK_DIM = 256;
} // namespace uzu::moe
