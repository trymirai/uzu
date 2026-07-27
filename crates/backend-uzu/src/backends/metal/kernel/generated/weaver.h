// Auto-generated from gpu_types/weaver - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::weaver {
static constant constexpr size_t CANDIDATES_MAX = 512;

static constant constexpr size_t TOP_CHILDREN_THREADS = 256;

static constant constexpr size_t TOP_CHILDREN_SIMDGROUPS = 8;

enum class FrontierIdx : uint32_t {
  TokenId = 0,
  ParentSlot = 1,
  Depth = 2,
  PathLogprobBits = 3,
  EdgeLogprobBits = 4,
  PathScoreKey = 5,
  Active = 6,
};

enum class TreeIdx : uint32_t {
  TokenId = 0,
  ParentSlot = 1,
  Depth = 2,
  PathLogprobBits = 3,
  EdgeLogprobBits = 4,
  Valid = 5,
};

enum class MetadataIdx : uint32_t {
  Depth = 0,
  AncestorCount = 1,
  TreeSlot = 2,
};

static constant constexpr uint32_t FRONTIER_NO_WINNER = ~0;

static constant constexpr size_t FRONTIER_SELECT_THREADS = 256;

static constant constexpr size_t FRONTIER_SELECT_SIMDGROUPS = 8;

static constant constexpr size_t FRONTIER_ENTRIES_PER_THREAD = 8;

static constant constexpr size_t FRONTIER_MAX_SLOTS = 2048;

static constant constexpr size_t FRONTIER_MAX_WIDTH = 32;
} // namespace uzu::weaver
