// Auto-generated from gpu_types/moe - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::moe {
enum class MoeExpertFormat : uint32_t {
  Nvfp4 = 0,
  AffineU4 = 1,
};

static constant constexpr uint32_t MOE_MAX_TOP_K = 8;

static constant constexpr uint32_t MOE_MAX_EXPERTS = 256;

static constant constexpr uint32_t MOE_GEMV_MAX_TOKENS = 32;

static constant constexpr uint32_t MOE_GEMV_SIMDGROUPS = 8;

static constant constexpr uint32_t MOE_UP_TILE_HIDDEN = 4;

static constant constexpr uint32_t MOE_DOWN_TILE_ROWS = 16;

static constant constexpr uint32_t MOE_ROUTE_EXPERTS_PER_GROUP = 4;

static constant constexpr uint32_t MOE_ROUTE_FINALIZE_THREADS = 1024;

static constant constexpr uint32_t MOE_ACTIVE_EXPERT_BITS = 8;

static constant constexpr uint32_t MOE_ACTIVE_COUNT_BITS = 11;
} // namespace uzu::moe
