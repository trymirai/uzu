// Auto-generated from gpu_types/confidence_head - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::confidence_head {
static constant constexpr uint32_t MODEL_MAX = 64;

static constant constexpr uint32_t POOL_HIDDEN_MAX = 16;

static constant constexpr uint32_t EDGE_HIDDEN_MAX = 128;

static constant constexpr uint32_t CHILDREN_MAX = 4;

static constant constexpr uint32_t SLOT_NUMBERS = 4;

static constant constexpr uint32_t POOL_FEATURES = 3;

static constant constexpr uint32_t POOL_INPUTS = 3;

static constant constexpr uint32_t RELATIONS = 6;

static constant constexpr uint32_t TREE_ROWS_MAX = 1024;

typedef struct {
  uint32_t candidates;
  uint32_t children;
  uint32_t tree_slot_count;
  uint32_t vocab_size;
  float logit_cap;
  float epsilon;
} ConfidenceHeadParams;
} // namespace uzu::confidence_head
