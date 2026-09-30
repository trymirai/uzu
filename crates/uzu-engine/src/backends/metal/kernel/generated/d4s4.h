// Auto-generated from gpu_types/d4s4 - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::d4s4 {
static constant constexpr uint32_t VALUES_PER_CODE = 4;

static constant constexpr uint32_t CODEBOOK_SIZE = 256;

static constant constexpr uint32_t COLUMNS_PER_LADDER_SCALE = 64;

static constant constexpr uint32_t COLUMNS_PER_LADDER_INDEX_BYTE = 2 * COLUMNS_PER_LADDER_SCALE;

static constant constexpr uint32_t LADDER_SIZE = 16;
} // namespace uzu::d4s4
