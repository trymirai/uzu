// Auto-generated from gpu_types/attention_prepare - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::attention_prepare {
typedef struct {
  uint32_t key_row_stride;
  uint32_t value_row_stride;
  float query_clip_threshold;
  float key_clip_threshold;
  float log_scaling_n_floor;
  float log_scaling_alpha;
  uint32_t position_offset;
} AttentionPrepareParams;

typedef struct {
  uint32_t query_row_stride;
  uint32_t key_row_stride;
  uint32_t value_row_stride;
  uint32_t relative_offset;
  uint32_t kv_token_offset;
  uint32_t query_rows;
  uint32_t position_offset;
  float query_clip_threshold;
  float key_clip_threshold;
  float relative_clip_threshold;
  float log_scaling_n_floor;
  float log_scaling_alpha;
  float key_conv_scale;
  float value_conv_scale;
  float value_norm_epsilon;
} SeparatePrepareParams;
} // namespace uzu::attention_prepare
