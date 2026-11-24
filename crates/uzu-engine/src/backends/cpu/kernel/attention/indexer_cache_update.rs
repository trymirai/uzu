use half::bf16;
use uzu_engine_macros::kernel;

#[kernel(IndexerCacheUpdate)]
pub fn indexer_cache_update(
    block_codes: *mut i8,
    block_scales: *mut bf16,
    unpooled_keys: *mut bf16,
    accepted_indices: &[u32],
    accepted_count: u32,
    prefix_tokens: u32,
    cached_block_row: u32,
    #[specialize] head_dim: u32,
    #[specialize] compression_ratio: u32,
) {
    let partial_block_start = (prefix_tokens + accepted_count) / compression_ratio * compression_ratio;
    for (index, &draft_node) in accepted_indices.iter().take(accepted_count as usize).enumerate() {
        let position = prefix_tokens + index as u32;
        unsafe {
            if (position + 1).is_multiple_of(compression_ratio) {
                let block_row = cached_block_row + position / compression_ratio;
                std::ptr::copy_nonoverlapping(
                    block_codes.add((draft_node * head_dim) as usize),
                    block_codes.add((block_row * head_dim) as usize),
                    head_dim as usize,
                );
                *block_scales.add(block_row as usize) = *block_scales.add(draft_node as usize);
            }
            if position >= partial_block_start {
                let tail_row = position % compression_ratio;
                let draft_row = compression_ratio - 1 + draft_node;
                std::ptr::copy_nonoverlapping(
                    unpooled_keys.add((draft_row * head_dim) as usize),
                    unpooled_keys.add((tail_row * head_dim) as usize),
                    head_dim as usize,
                );
            }
        }
    }
}
