use proc_macros::kernel;

// Gram fused with the former ScaleQk pass: emits the per-k-head kk block and,
// for each of the k-head's GQA v-heads, the causal-masked decay-scaled qk block
// qk_scaled[row, col] = qk * exp(g_row - g_col) (col <= row, else 0).
#[kernel(DeltaNetChunkedGram)]
#[variants(HEAD_K_DIM, 128)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
#[variants(USE_MXU, false, true)]
pub fn delta_net_chunked_gram<const HEAD_K_DIM: u32, const CHUNK_SIZE: u32, const USE_MXU: bool>(
    q_norm: *const f32,
    k_norm: *const f32,
    g: *const f32,
    kk_out: *mut f32,
    qk_scaled_out: *mut f32,
    num_v_heads: u32,
    num_k_heads: u32,
    key_dim: u32,
    suffix_len: u32,
) {
    let _ = USE_MXU;
    let head_k_dim = HEAD_K_DIM as usize;
    let chunk_size = CHUNK_SIZE as usize;
    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let key_dim = key_dim as usize;
    let suffix_len = suffix_len as usize;
    let num_chunks = suffix_len.div_ceil(chunk_size);
    let groups_per_head = num_v_heads / num_k_heads;

    for chunk in 0..num_chunks {
        let chunk_token = chunk * chunk_size;
        let valid_tokens = suffix_len.saturating_sub(chunk_token).min(chunk_size);
        for hk in 0..num_k_heads {
            let kk_base = (chunk * num_k_heads + hk) * chunk_size * chunk_size;
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
                    unsafe { *kk_out.add(kk_base + row * chunk_size + col) = kk };

                    // Expand qk to each GQA v-head with causal mask + decay scale.
                    for group in 0..groups_per_head {
                        let hv = hk * groups_per_head + group;
                        let dst = (chunk * num_v_heads + hv) * chunk_size * chunk_size + row * chunk_size + col;
                        let scaled = if row >= valid_tokens || col >= valid_tokens || col > row {
                            0.0
                        } else {
                            let g_row = unsafe { *g.add((chunk_token + row) * num_v_heads + hv) };
                            let g_col = unsafe { *g.add((chunk_token + col) * num_v_heads + hv) };
                            qk * (g_row - g_col).exp()
                        };
                        unsafe { *qk_scaled_out.add(dst) = scaled };
                    }
                }
            }
        }
    }
}
