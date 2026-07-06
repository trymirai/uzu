use half::{bf16, f16};
use num_traits::{Float, NumCast};
use proc_macros::kernel;

use crate::array::ArrayElement;

// CPU mirror of the fused persistent chunk-scan kernel `DeltaNetChunkedFusedApply`.
// The Metal kernel partitions work across (head, value-slice) threadgroups and
// keeps the state transposed in threadgroup memory; the math per value column is
// identical and independent of the tiling, so this reference ignores VT and walks
// every head and chunk serially, mutating `state` in place. State is f32
// end-to-end; W/U are bf16 (matching the Metal precision contract).
#[kernel(DeltaNetChunkedFusedApply)]
#[variants(T, f32, f16, bf16)]
#[variants(VT, 16, 32)]
pub fn delta_net_chunked_fused_apply<T: ArrayElement + Float, const VT: u32>(
    w: *const bf16,
    u: *const bf16,
    q_norm: *const f32,
    k_norm: *const f32,
    qk_scaled: *const f32,
    log_decay: *const f32,
    state: *mut f32,
    out: *mut T,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    key_dim: u32,
    value_dim: u32,
    suffix_len: u32,
) {
    let _ = VT;
    const HEAD_K_DIM: usize = 128;
    const CHUNK_SIZE: usize = 64;

    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let head_v_dim = head_v_dim as usize;
    let key_dim = key_dim as usize;
    let value_dim = value_dim as usize;
    let suffix_len = suffix_len as usize;
    let num_chunks = suffix_len.div_ceil(CHUNK_SIZE);
    let groups_per_head = num_v_heads / num_k_heads;

    let mut vnew = vec![0.0f32; CHUNK_SIZE * head_v_dim];

    for hv in 0..num_v_heads {
        let hk = hv / groups_per_head;
        for chunk in 0..num_chunks {
            let token_base = chunk * CHUNK_SIZE;
            let valid_tokens = suffix_len.saturating_sub(token_base).min(CHUNK_SIZE);
            if valid_tokens == 0 {
                continue;
            }
            let chunk_head_base = chunk * num_v_heads + hv;

            // Chunk-local prefix g[local_t] = sum_{i<=local_t} log_decay
            // (replaces the former Cumsum dispatch + g buffer). Only valid tokens
            // are summed; entries >= valid_tokens are unused.
            let mut g_local = [0.0f32; CHUNK_SIZE];
            let mut g_acc = 0.0f32;
            for (i, slot) in g_local.iter_mut().enumerate().take(valid_tokens) {
                g_acc += unsafe { *log_decay.add((token_base + i) * num_v_heads + hv) };
                *slot = g_acc;
            }

            // Phase 1: Vnew = U - W . S^T (S = state before this chunk).
            for local_t in 0..CHUNK_SIZE {
                let wu_row = (chunk_head_base * CHUNK_SIZE + local_t) * head_v_dim;
                if local_t >= valid_tokens {
                    for dv in 0..head_v_dim {
                        vnew[local_t * head_v_dim + dv] = 0.0;
                    }
                    continue;
                }
                let w_row = (chunk_head_base * CHUNK_SIZE + local_t) * HEAD_K_DIM;
                for dv in 0..head_v_dim {
                    let state_row = (hv * head_v_dim + dv) * HEAD_K_DIM;
                    let mut correction = 0.0f32;
                    for dk in 0..HEAD_K_DIM {
                        let w_value: f32 = NumCast::from(unsafe { *w.add(w_row + dk) }).unwrap();
                        correction += w_value * unsafe { *state.add(state_row + dk) };
                    }
                    let u_value: f32 = NumCast::from(unsafe { *u.add(wu_row + dv) }).unwrap();
                    vnew[local_t * head_v_dim + dv] = u_value - correction;
                }
            }

            // Phase 2: Y = exp(g) (.) (Q . S^T) + A . Vnew -> out.
            for local_t in 0..valid_tokens {
                let token = token_base + local_t;
                let g_scale = g_local[local_t].exp();
                let q_base = token * key_dim + hk * HEAD_K_DIM;
                let qk_base = chunk_head_base * CHUNK_SIZE * CHUNK_SIZE + local_t * CHUNK_SIZE;
                for dv in 0..head_v_dim {
                    let state_row = (hv * head_v_dim + dv) * HEAD_K_DIM;
                    let mut value = 0.0f32;
                    for dk in 0..HEAD_K_DIM {
                        value += unsafe { *q_norm.add(q_base + dk) } * unsafe { *state.add(state_row + dk) };
                    }
                    value *= g_scale;
                    // A (qk_scaled) is causal-masked, so summing the full chunk
                    // matches the tiled Metal path.
                    for j in 0..valid_tokens {
                        let a = unsafe { *qk_scaled.add(qk_base + j) };
                        value += a * vnew[j * head_v_dim + dv];
                    }
                    unsafe { *out.add(token * value_dim + hv * head_v_dim + dv) = T::from(value).unwrap() };
                }
            }

            // Phase 3: S^T <- alpha . S^T + (decay_scale (.) K)^T . Vnew, where
            // decay_scale[t] = exp(g_last - g_t) is folded in on the fly (this
            // replaces the separate DecayScale dispatch; beta is already baked
            // into Vnew via U/W, so it does not appear here).
            let g_last = g_local[valid_tokens - 1];
            let alpha = g_last.exp();
            for dv in 0..head_v_dim {
                let state_row = (hv * head_v_dim + dv) * HEAD_K_DIM;
                for dk in 0..HEAD_K_DIM {
                    let mut value = alpha * unsafe { *state.add(state_row + dk) };
                    for local_t in 0..valid_tokens {
                        let token = token_base + local_t;
                        let k = unsafe { *k_norm.add(token * key_dim + hk * HEAD_K_DIM + dk) };
                        let decay = (g_last - g_local[local_t]).exp();
                        value += vnew[local_t * head_v_dim + dv] * k * decay;
                    }
                    unsafe { *state.add(state_row + dk) = value };
                }
            }
        }
    }
}
