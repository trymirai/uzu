use proc_macros::kernel;

#[kernel(DeltaNetChunkedStateA2DecayScale)]
#[variants(HEAD_K_DIM, 128)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
pub fn delta_net_chunked_state_a2_decay_scale<const HEAD_K_DIM: u32, const CHUNK_SIZE: u32>(
    g: *const f32,
    decay_scale: *mut f32,
    num_v_heads: u32,
    suffix_len: u32,
) {
    let chunk_size = CHUNK_SIZE as usize;
    let num_v_heads = num_v_heads as usize;
    let suffix_len = suffix_len as usize;
    let num_chunks = suffix_len.div_ceil(chunk_size);

    for chunk in 0..num_chunks {
        let token_base = chunk * chunk_size;
        let valid_tokens = suffix_len.saturating_sub(token_base).min(chunk_size);
        if valid_tokens == 0 {
            continue;
        }
        for hv in 0..num_v_heads {
            let g_last = unsafe { *g.add((token_base + valid_tokens - 1) * num_v_heads + hv) };
            for local_t in 0..chunk_size {
                let dst = (chunk * num_v_heads + hv) * chunk_size + local_t;
                if local_t >= valid_tokens {
                    unsafe { *decay_scale.add(dst) = 0.0 };
                    continue;
                }
                let token = token_base + local_t;
                unsafe { *decay_scale.add(dst) = (g_last - *g.add(token * num_v_heads + hv)).exp() };
            }
        }
    }
}
