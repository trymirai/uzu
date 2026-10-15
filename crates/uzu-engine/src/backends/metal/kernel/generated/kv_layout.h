// Auto-generated from gpu_types/kv_layout - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::kv_layout {
static constant constexpr uint32_t TILE_KEYS = 32;

static constant constexpr uint32_t TILE_DIMS = 16;

static constant constexpr uint32_t FRAGMENT_BYTES = TILE_KEYS * TILE_DIMS;

static constant constexpr uint32_t SLICE_DIMS = 64;

static constant constexpr int16_t V_CODE_BIAS = 128;

static constant constexpr uint32_t GROUP_ROWS = 8;

static constant constexpr uint32_t LANE_COLS = 4;

static constant constexpr uint32_t LANE_BYTES = TILE_KEYS / GROUP_ROWS * LANE_COLS;

static constant constexpr uint32_t WORD_BITS = 32;

static constant constexpr uint32_t ROTATION_SIGN_BITS[8] =
    {2654435769, 2246822507, 3266489909, 668265263, 374761393, 3210233709, 2496678331, 3605593784};
} // namespace uzu::kv_layout
