// Auto-generated from gpu_types/router_topk - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::router_topk {
static constant constexpr uint32_t ROUTER_TOPK_MAX_EXPERTS = 512;

static constant constexpr uint32_t ROUTER_TOPK_MAX_SELECTED_EXPERTS = 128;

static constant constexpr uint32_t ROUTER_TOPK_MAX_MODEL_DIM = 4096;
} // namespace uzu::router_topk
