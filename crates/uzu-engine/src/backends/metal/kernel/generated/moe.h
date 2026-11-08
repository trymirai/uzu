// Auto-generated from gpu_types/moe - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::moe {
static constant constexpr uint32_t ROUTER_MAX_TOP_K = 8;

static constant constexpr uint32_t ROUTER_MAX_EXPERTS = 256;

static constant constexpr uint32_t GEMV_GROUP_SIZE = 64;

static constant constexpr uint32_t GEMV_W4_INPUT_BLOCK = 512;

static constant constexpr uint32_t GEMV_W8_INPUT_BLOCK = 256;

static constant constexpr uint32_t GEMV_OUTPUT_ROW_TILE = 32;

static constant constexpr uint32_t COMBINE_BLOCK_DIM = 256;
} // namespace uzu::moe
