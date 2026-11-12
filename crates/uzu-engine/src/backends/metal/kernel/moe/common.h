#pragma once

#include <metal_stdlib>
#include "../generated/moe.h"

#define MOE_ROUTER_MAX_PER_LANE ((uzu::moe::ROUTER_MAX_EXPERTS + METAL_SIMD_SIZE - 1) / METAL_SIMD_SIZE)
#define MOE_GEMV_NUM_SIMDGROUPS 8
#define MOE_GEMV_OUTPUT_ROW_TILE 32
