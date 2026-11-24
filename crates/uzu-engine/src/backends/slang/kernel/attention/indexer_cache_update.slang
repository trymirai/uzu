import definitions;

[[Public]] [[Kernel]] func IndexerCacheUpdate(
  RWStructuredBuffer<int8_t> block_codes,
  RWStructuredBuffer<BFloat16> block_scales,
  RWStructuredBuffer<BFloat16> unpooled_keys,
  [[Constant]] StructuredBuffer<uint> accepted_indices,
  uint32_t accepted_count,
  uint32_t prefix_tokens,
  uint32_t cached_block_row,
  [[Specialize]] uint32_t head_dim,
  [[Specialize]] uint32_t compression_ratio,
  [[Axis("accepted_count * head_dim", "256")]] uint32_t element,
) {
  // TODO
}
