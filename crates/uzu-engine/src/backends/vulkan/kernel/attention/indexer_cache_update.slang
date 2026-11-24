import definitions;

[[Kernel]] [[Public]]
[[Precondition("new", "head_dim > 0 && compression_ratio > 0")]]
[[Precondition("encode", "accepted_count as usize <= accepted_indices.len()")]]
func IndexerCacheUpdate(
  Ptr<int8_t, Access.ReadWrite> block_codes,
  Ptr<bf16, Access.ReadWrite> block_scales,
  Ptr<bf16, Access.ReadWrite> unpooled_keys,
  [[Constant]] Ptr<uint, Access.Read> accepted_indices,
  uint accepted_count,
  uint prefix_tokens,
  uint cached_block_row,
  [[Specialize]] uint head_dim,
  [[Specialize]] uint compression_ratio,
  [[Axis("accepted_count * head_dim", "256")]] uint element,
) {
  let accepted_index = element / head_dim;
  let channel = element % head_dim;
  let draft_node = accepted_indices[accepted_index];
  let position = prefix_tokens + accepted_index;
  if ((position + 1) % compression_ratio == 0) {
    let block_row = cached_block_row + position / compression_ratio;
    block_codes[block_row * head_dim + channel] = block_codes[draft_node * head_dim + channel];
    if (channel == 0) {
      block_scales[block_row] = block_scales[draft_node];
    }
  }
  let partial_block_start = (prefix_tokens + accepted_count) / compression_ratio * compression_ratio;
  if (position >= partial_block_start) {
    let tail_row = position % compression_ratio;
    let draft_row = compression_ratio - 1 + draft_node;
    unpooled_keys[tail_row * head_dim + channel] = unpooled_keys[draft_row * head_dim + channel];
  }
}
