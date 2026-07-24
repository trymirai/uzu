// Auto-generated from gpu_types/weaver - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::weaver {
static constant constexpr size_t CANDIDATES_MAX = 512;

static constant constexpr size_t TOP_CHILDREN_THREADS = 256;

static constant constexpr size_t TOP_CHILDREN_SIMDGROUPS = 8;

static constant constexpr size_t FRONTIER_LANE_TOKEN = 0;

static constant constexpr size_t FRONTIER_LANE_PARENT = 1;

static constant constexpr size_t FRONTIER_LANE_DEPTH = 2;

static constant constexpr size_t FRONTIER_LANE_CUM = 3;

static constant constexpr size_t FRONTIER_LANE_LOGPROB = 4;

static constant constexpr size_t FRONTIER_LANE_KEY = 5;

static constant constexpr size_t FRONTIER_LANE_ACTIVE = 6;

static constant constexpr size_t FRONTIER_LANE_COUNT = FRONTIER_LANE_ACTIVE + 1;

static constant constexpr size_t TREE_LANE_TOKEN = 0;

static constant constexpr size_t TREE_LANE_PARENT = 1;

static constant constexpr size_t TREE_LANE_DEPTH = 2;

static constant constexpr size_t TREE_LANE_CUM = 3;

static constant constexpr size_t TREE_LANE_LOGPROB = 4;

static constant constexpr size_t TREE_LANE_MASK = 5;

static constant constexpr size_t TREE_LANE_COUNT = TREE_LANE_MASK + 1;

static constant constexpr size_t METADATA_LANE_DEPTH = 0;

static constant constexpr size_t METADATA_LANE_ANCESTOR_COUNT = 1;

static constant constexpr size_t METADATA_LANE_NODE_INDEX = 2;

static constant constexpr size_t METADATA_LANE_COUNT = METADATA_LANE_NODE_INDEX + 1;

static constant constexpr uint32_t FRONTIER_NO_WINNER = ~0;

static constant constexpr size_t FRONTIER_SELECT_THREADS = 256;

static constant constexpr size_t FRONTIER_SELECT_SIMDGROUPS = 8;

static constant constexpr size_t FRONTIER_ENTRIES_PER_THREAD = 8;

static constant constexpr size_t FRONTIER_MAX_SLOTS = 2048;

static constant constexpr size_t FRONTIER_MAX_WIDTH = 32;
} // namespace uzu::weaver
