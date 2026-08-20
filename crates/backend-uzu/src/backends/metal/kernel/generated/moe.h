// Auto-generated from gpu_types/moe - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::moe {
static constant constexpr uint32_t ROUTER_MAX_EXPERTS = 512;

static constant constexpr uint32_t ROUTER_MAX_SELECTED_EXPERTS = 128;

static constant constexpr uint32_t ROUTER_MAX_MODEL_DIM = 4096;

static constant constexpr uint32_t ROUTER_THREADS_PER_THREADGROUP = 256;
} // namespace uzu::moe
