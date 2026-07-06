use half::bf16;
use proc_macros::kernel;

// CPU mirror of `DeltaNetChunkedSolveT`: emits the dense unit-lower-triangular
// inverse T = (I + A)^{-1} per (chunk, v-head) as BF16, where
// A[i,k] = beta_i * exp(g_i - g_k) * kk[i,k] for k < i (else 0). The forward
// substitution accumulates through the BF16 intermediate exactly as the Metal
// kernel does (matching the old W/U precision contract).
#[kernel(DeltaNetChunkedSolveT)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
pub fn delta_net_chunked_solve_t<const CHUNK_SIZE: u32>(
    kk: *const f32,
    beta: *const f32,
    g: *const f32,
    t_out: *mut bf16,
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

    let mut a = vec![0.0f32; chunk_size * chunk_size];
    let mut t = vec![bf16::from_f32(0.0); chunk_size * chunk_size];

    for chunk in 0..num_chunks {
        let token_base = chunk * chunk_size;
        let valid = suffix_len.saturating_sub(token_base).min(chunk_size);
        for hv in 0..num_v_heads {
            let hk = hv / groups_per_head;
            let kk_base = (chunk * num_k_heads + hk) * chunk_size * chunk_size;

            for i in 0..chunk_size {
                for k in 0..chunk_size {
                    let value = if k < i && i < valid && k < valid {
                        let beta_i = unsafe { *beta.add((token_base + i) * num_v_heads + hv) };
                        let g_i = unsafe { *g.add((token_base + i) * num_v_heads + hv) };
                        let g_k = unsafe { *g.add((token_base + k) * num_v_heads + hv) };
                        let kk_value = unsafe { *kk.add(kk_base + i * chunk_size + k) };
                        beta_i * (g_i - g_k).exp() * kk_value
                    } else {
                        0.0
                    };
                    a[i * chunk_size + k] = value;
                    t[i * chunk_size + k] = bf16::from_f32(0.0);
                }
            }

            for j in 0..chunk_size {
                for i in j..chunk_size {
                    let mut acc = if i == j {
                        1.0f32
                    } else {
                        0.0f32
                    };
                    for k in j..i {
                        acc -= a[i * chunk_size + k] * t[k * chunk_size + j].to_f32();
                    }
                    t[i * chunk_size + j] = bf16::from_f32(acc);
                }
            }

            let t_base = (chunk * num_v_heads + hv) * chunk_size * chunk_size;
            for idx in 0..chunk_size * chunk_size {
                unsafe { *t_out.add(t_base + idx) = t[idx] };
            }
        }
    }
}
