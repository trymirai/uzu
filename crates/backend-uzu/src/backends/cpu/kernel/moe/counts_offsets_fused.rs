use proc_macros::kernel;

#[kernel(MoeCountsOffsetsFused)]
pub fn moe_counts_offsets_fused(
    selected_expert_ids: *const i32,
    expert_offsets: *mut u32,
    total_routed_row_count: *mut u32,
    scatter_block_expert_counts: *mut u32,
    token_count: u32,
    expert_count: u32,
    selected_experts_per_token: u32,
) {
    let expert_count = expert_count as usize;
    let token_count = token_count as usize;
    let selected_experts_per_token = selected_experts_per_token as usize;

    if expert_count == 0 {
        unsafe {
            *expert_offsets = 0;
            *total_routed_row_count = 0;
        }
        return;
    }

    const SCATTER_BLOCK_SIZE: usize = 256;
    const EXPERT_TILE_SIZE: usize = 512;
    let scatter_block_count = token_count.div_ceil(SCATTER_BLOCK_SIZE);
    let expert_tile_count = expert_count.div_ceil(EXPERT_TILE_SIZE);

    // Phase 1: Count tokens per expert and per scatter block.
    let mut total_expert_counts = vec![0u32; expert_count];
    for scatter_block_index in 0..scatter_block_count {
        let mut expert_counts_in_block = vec![0u32; expert_count];
        let token_block_start = scatter_block_index * SCATTER_BLOCK_SIZE;
        let token_block_end = (token_block_start + SCATTER_BLOCK_SIZE).min(token_count);
        for token_index in token_block_start..token_block_end {
            let selected_expert_row_start = token_index * selected_experts_per_token;
            for selected_expert_slot in 0..selected_experts_per_token {
                let selected_expert_id =
                    unsafe { *selected_expert_ids.add(selected_expert_row_start + selected_expert_slot) };
                if selected_expert_id >= 0 {
                    let expert_index = selected_expert_id as usize;
                    if expert_index < expert_count {
                        total_expert_counts[expert_index] += 1;
                        expert_counts_in_block[expert_index] += 1;
                    }
                }
            }
        }

        for (expert_index, expert_count_in_block) in expert_counts_in_block.into_iter().enumerate() {
            let expert_tile_index = expert_index / EXPERT_TILE_SIZE;
            let expert_index_in_tile = expert_index % EXPERT_TILE_SIZE;
            let partial_count_index =
                (scatter_block_index * expert_tile_count + expert_tile_index) * EXPERT_TILE_SIZE + expert_index_in_tile;
            unsafe { *scatter_block_expert_counts.add(partial_count_index) = expert_count_in_block };
        }
    }

    // Phase 2: Exclusive prefix scan to produce offsets
    let mut expert_offset_carry = 0u32;
    for expert_index in 0..expert_count {
        unsafe { *expert_offsets.add(expert_index) = expert_offset_carry };
        expert_offset_carry += total_expert_counts[expert_index];
    }
    unsafe {
        *expert_offsets.add(expert_count) = expert_offset_carry;
        *total_routed_row_count = expert_offset_carry;
    }
}
