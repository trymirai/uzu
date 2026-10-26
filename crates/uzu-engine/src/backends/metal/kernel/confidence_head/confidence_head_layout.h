#pragma once

#include <metal_stdlib>

using namespace metal;

#define CH_NODE_OUTPUTS 3u
#define CH_POOL_INPUTS 3u
#define CH_SLOT 4u

// Offsets, in floats, of the local part's weights in the packed buffer: the same order as
// encodable_block::conf_head::layout::LocalLayout.
struct ConfidenceHeadLayout {
  uint pool_entry_weights;
  uint pool_entry_biases;
  uint pool_hidden_weights;
  uint pool_hidden_biases;
  uint pool_out_weights;
  uint pool_out_biases;
  uint slot_means;
  uint slot_stds;
  uint slot_keep;
  uint edge_input_scales;
  uint edge_input_biases;
  uint edge_entry_weights;
  uint edge_entry_biases;
  uint edge_block_scales;
  uint edge_block_biases;
  uint edge_up_weights;
  uint edge_up_biases;
  uint edge_down_weights;
  uint edge_down_biases;
  uint edge_final_scales;
  uint edge_final_biases;
  uint delta_weights;
  uint delta_bias;
  uint output_weights;
  uint output_biases;
};

METAL_FUNC uint confidence_head_edge_input(uint hidden) { return 3u * hidden + CH_SLOT; }

METAL_FUNC ConfidenceHeadLayout confidence_head_layout(uint hidden, uint pool_hidden, uint edge_hidden) {
  const uint edge_input = confidence_head_edge_input(hidden);
  ConfidenceHeadLayout layout;
  uint offset = 0u;
  layout.pool_entry_weights = offset;
  offset += pool_hidden * CH_POOL_INPUTS;
  layout.pool_entry_biases = offset;
  offset += pool_hidden;
  layout.pool_hidden_weights = offset;
  offset += pool_hidden * pool_hidden;
  layout.pool_hidden_biases = offset;
  offset += pool_hidden;
  layout.pool_out_weights = offset;
  offset += hidden * 2u * pool_hidden;
  layout.pool_out_biases = offset;
  offset += hidden;
  layout.slot_means = offset;
  offset += CH_SLOT;
  layout.slot_stds = offset;
  offset += CH_SLOT;
  layout.slot_keep = offset;
  offset += CH_SLOT;
  layout.edge_input_scales = offset;
  offset += edge_input;
  layout.edge_input_biases = offset;
  offset += edge_input;
  layout.edge_entry_weights = offset;
  offset += hidden * edge_input;
  layout.edge_entry_biases = offset;
  offset += hidden;
  layout.edge_block_scales = offset;
  offset += hidden;
  layout.edge_block_biases = offset;
  offset += hidden;
  layout.edge_up_weights = offset;
  offset += edge_hidden * hidden;
  layout.edge_up_biases = offset;
  offset += edge_hidden;
  layout.edge_down_weights = offset;
  offset += hidden * edge_hidden;
  layout.edge_down_biases = offset;
  offset += hidden;
  layout.edge_final_scales = offset;
  offset += hidden;
  layout.edge_final_biases = offset;
  offset += hidden;
  layout.delta_weights = offset;
  offset += hidden;
  layout.delta_bias = offset;
  offset += 1u;
  layout.output_weights = offset;
  offset += CH_NODE_OUTPUTS * hidden;
  layout.output_biases = offset;
  return layout;
}
