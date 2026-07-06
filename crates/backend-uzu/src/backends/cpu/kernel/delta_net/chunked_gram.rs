use proc_macros::kernel;

#[kernel(DeltaNetChunkedGram)]
#[variants(HEAD_K_DIM, 128)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
pub fn delta_net_chunked_gram<const HEAD_K_DIM: u32, const CHUNK_SIZE: u32>(
    q_norm: *const f32,
    k_norm: *const f32,
    kk_out: *mut f32,
    qk_out: *mut f32,
    num_k_heads: u32,
    key_dim: u32,
    suffix_len: u32,
) {
    let head_k_dim = HEAD_K_DIM as usize;
    let chunk_size = CHUNK_SIZE as usize;
    let num_k_heads = num_k_heads as usize;
    let key_dim = key_dim as usize;
    let suffix_len = suffix_len as usize;
    let num_chunks = suffix_len.div_ceil(chunk_size);

    for chunk in 0..num_chunks {
        let chunk_token = chunk * chunk_size;
        for hk in 0..num_k_heads {
            let out_base = (chunk * num_k_heads + hk) * chunk_size * chunk_size;
            for row in 0..chunk_size {
                let row_token = chunk_token + row;
                for col in 0..chunk_size {
                    let col_token = chunk_token + col;
                    let mut kk = 0.0f32;
                    let mut qk = 0.0f32;
                    if row_token < suffix_len && col_token < suffix_len {
                        let q_base = row_token * key_dim + hk * head_k_dim;
                        let k_row_base = row_token * key_dim + hk * head_k_dim;
                        let k_col_base = col_token * key_dim + hk * head_k_dim;
                        for dim in 0..head_k_dim {
                            let k_col = unsafe { *k_norm.add(k_col_base + dim) };
                            kk += unsafe { *k_norm.add(k_row_base + dim) } * k_col;
                            qk += unsafe { *q_norm.add(q_base + dim) } * k_col;
                        }
                    }
                    unsafe {
                        *kk_out.add(out_base + row * chunk_size + col) = kk;
                        *qk_out.add(out_base + row * chunk_size + col) = qk;
                    }
                }
            }
        }
    }
}

#[kernel(DeltaNetChunkedScaleQk)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
pub fn delta_net_chunked_scale_qk<const CHUNK_SIZE: u32>(
    qk: *const f32,
    g: *const f32,
    qk_scaled: *mut f32,
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

    for chunk in 0..num_chunks {
        let token_base = chunk * chunk_size;
        let valid_tokens = suffix_len.saturating_sub(token_base).min(chunk_size);
        for hv in 0..num_v_heads {
            let hk = hv / groups_per_head;
            let src_base = (chunk * num_k_heads + hk) * chunk_size * chunk_size;
            let dst_base = (chunk * num_v_heads + hv) * chunk_size * chunk_size;
            for row in 0..chunk_size {
                for col in 0..chunk_size {
                    let dst = dst_base + row * chunk_size + col;
                    if row >= valid_tokens || col >= valid_tokens || col > row {
                        unsafe { *qk_scaled.add(dst) = 0.0 };
                        continue;
                    }
                    let g_row = unsafe { *g.add((token_base + row) * num_v_heads + hv) };
                    let g_col = unsafe { *g.add((token_base + col) * num_v_heads + hv) };
                    unsafe {
                        *qk_scaled.add(dst) = *qk.add(src_base + row * chunk_size + col) * (g_row - g_col).exp();
                    }
                }
            }
        }
    }
}
