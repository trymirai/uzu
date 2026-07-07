use proc_macros::kernel;

fn chunked_g<const RECOMPUTE_G: bool>(
    g_or_log_decay: *const f32,
    num_v_heads: usize,
    hv: usize,
    token_base: usize,
    local_t: usize,
) -> f32 {
    if RECOMPUTE_G {
        let mut acc = 0.0f32;
        for i in 0..=local_t {
            acc += unsafe { *g_or_log_decay.add((token_base + i) * num_v_heads + hv) };
        }
        acc
    } else {
        unsafe { *g_or_log_decay.add((token_base + local_t) * num_v_heads + hv) }
    }
}

#[kernel(DeltaNetChunkedSolve)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
#[variants(RECOMPUTE_G, false, true)]
pub fn delta_net_chunked_solve<const CHUNK_SIZE: u32, const RECOMPUTE_G: bool>(
    kk: *const f32,
    beta: *const f32,
    g_or_log_decay: *const f32,
    a_packed: *mut f32,
    a_inv: *mut f32,
    num_v_heads: u32,
    num_k_heads: u32,
    suffix_len: u32,
) {
    let chunk_size = CHUNK_SIZE as usize;
    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let suffix_len = suffix_len as usize;
    let num_chunks = suffix_len.div_ceil(chunk_size);
    let groups_per_head = num_v_heads / num_k_heads;
    let block_size = 16usize;
    let num_blocks = chunk_size.div_ceil(block_size);
    let num_col_pairs = num_blocks.div_ceil(2);

    for chunk in 0..num_chunks {
        let token_base = chunk * chunk_size;
        for hv in 0..num_v_heads {
            let hk = hv / groups_per_head;
            for block in 0..num_blocks {
                for pair in 0..num_col_pairs {
                    for local_row in 0..block_size {
                        let row = block * block_size + local_row;
                        for local_col in 0..2 * block_size {
                            let col = pair * 2 * block_size + local_col;
                            let row_token = token_base + row;
                            let col_token = token_base + col;
                            let value = if row < chunk_size
                                && col < chunk_size
                                && row_token < suffix_len
                                && col_token < suffix_len
                                && col < row
                            {
                                let beta_row = unsafe { *beta.add(row_token * num_v_heads + hv) };
                                let g_row = chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, row);
                                let g_col = chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, col);
                                let kk_value = unsafe {
                                    *kk.add(
                                        (chunk * num_k_heads + hk) * chunk_size * chunk_size + row * chunk_size + col,
                                    )
                                };
                                beta_row * (g_row - g_col).exp() * kk_value
                            } else {
                                0.0
                            };
                            let out_idx = (((chunk * num_v_heads + hv) * num_blocks + block) * num_col_pairs + pair)
                                * (block_size * 2 * block_size)
                                + local_row * (2 * block_size)
                                + local_col;
                            unsafe { *a_packed.add(out_idx) = value };
                        }
                    }
                }

                for inv_col in 0..block_size {
                    let mut inverse_col = [0.0f32; 16];
                    inverse_col[inv_col] = 1.0;
                    for inv_row in 0..block_size {
                        let row = block * block_size + inv_row;
                        if inv_row > inv_col && row < chunk_size && token_base + row < suffix_len {
                            let mut acc = 0.0f32;
                            for prev in 0..inv_row {
                                let prev_col = block * block_size + prev;
                                let beta_row = unsafe { *beta.add((token_base + row) * num_v_heads + hv) };
                                let g_row = chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, row);
                                let g_prev =
                                    chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, prev_col);
                                let kk_value = unsafe {
                                    *kk.add(
                                        (chunk * num_k_heads + hk) * chunk_size * chunk_size
                                            + row * chunk_size
                                            + prev_col,
                                    )
                                };
                                let a = beta_row * (g_row - g_prev).exp() * kk_value;
                                acc += a * inverse_col[prev];
                            }
                            inverse_col[inv_row] = -acc;
                        }
                    }
                    for inv_row in 0..block_size {
                        let out_idx = ((chunk * num_v_heads + hv) * num_blocks + block) * block_size * block_size
                            + inv_row * block_size
                            + inv_col;
                        unsafe { *a_inv.add(out_idx) = inverse_col[inv_row] };
                    }
                }
            }
        }
    }
}
