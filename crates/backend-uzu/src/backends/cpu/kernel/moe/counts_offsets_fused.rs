use proc_macros::kernel;

#[kernel(MoeCountsOffsetsFused)]
pub fn moe_counts_offsets_fused(
    topk_ids: *const i32,
    offsets: *mut u32,
    sum_k_out: *mut u32,
    partials: *mut u32,
    t_input: u32,
    e_input: u32,
    k_input: u32,
) {
    let e = e_input as usize;
    let t = t_input as usize;
    let k = k_input as usize;

    if e == 0 {
        unsafe {
            *offsets = 0;
            *sum_k_out = 0;
        }
        return;
    }

    const SCATTER_BLOCK_SIZE: usize = 256;
    const TILE_E: usize = 512;
    let num_blocks = t.div_ceil(SCATTER_BLOCK_SIZE);
    let num_tiles = e.div_ceil(TILE_E);

    // Phase 1: Count tokens per expert and per scatter block.
    let mut counts = vec![0u32; e];
    for block_id in 0..num_blocks {
        let mut block_counts = vec![0u32; e];
        let t_start = block_id * SCATTER_BLOCK_SIZE;
        let t_end = (t_start + SCATTER_BLOCK_SIZE).min(t);
        for ti in t_start..t_end {
            let base = ti * k;
            for kk in 0..k {
                let eid = unsafe { *topk_ids.add(base + kk) };
                if eid >= 0 {
                    let ue = eid as usize;
                    if ue < e {
                        counts[ue] += 1;
                        block_counts[ue] += 1;
                    }
                }
            }
        }

        for (expert_id, count) in block_counts.into_iter().enumerate() {
            let tile_id = expert_id / TILE_E;
            let tile_expert_id = expert_id % TILE_E;
            let partial_idx = (block_id * num_tiles + tile_id) * TILE_E + tile_expert_id;
            unsafe { *partials.add(partial_idx) = count };
        }
    }

    // Phase 2: Exclusive prefix scan to produce offsets
    let mut sum = 0u32;
    for i in 0..e {
        unsafe { *offsets.add(i) = sum };
        sum += counts[i];
    }
    unsafe {
        *offsets.add(e) = sum;
        *sum_k_out = sum;
    }
}
