#include <metal_stdlib>
#include "../common/dsl.h"
#include "cache_copy.h"

namespace kv_layout = uzu::kv_layout;
namespace kv_cache = uzu::kv_cache;

PUBLIC KERNEL(IndexerCacheUpdate)(
    device int8_t* block_codes,
    device bfloat* block_scales,
    device bfloat* unpooled_keys,
    const constant uint* accepted_indices,
    const constant uint& accepted_count,
    const constant uint& prefix_tokens,
    const constant uint& cached_block_row,
    const uint head_dim SPECIALIZE,
    const uint compression_ratio SPECIALIZE,
    const uint element AXIS(accepted_count* head_dim, 256)
) {
  const uint accepted_index = element / head_dim;
  const uint channel = element % head_dim;
  const uint draft_node = accepted_indices[accepted_index];
  const uint token_position = prefix_tokens + accepted_index;

  // Draft rows and accepted rows occupy separate sections.
  if ((token_position + 1) % compression_ratio == 0) {
    const kv_layout::KVShape shape{head_dim, 1, kv_layout::KVLayout::RowMajor};
    const uint block_row = cached_block_row + token_position / compression_ratio;
    const auto key = kv_cache::load_key(block_codes, block_scales, shape, draft_node, 0, channel);
    kv_cache::store_key(block_codes, block_scales, shape, block_row, 0, channel, key);
  }

  const uint partial_block_start = (prefix_tokens + accepted_count) / compression_ratio * compression_ratio;
  if (token_position >= partial_block_start) {
    const uint tail_row = token_position % compression_ratio;
    const uint draft_row = compression_ratio - 1 + draft_node;
    unpooled_keys[size_t(tail_row) * head_dim + channel] = unpooled_keys[size_t(draft_row) * head_dim + channel];
  }
}
