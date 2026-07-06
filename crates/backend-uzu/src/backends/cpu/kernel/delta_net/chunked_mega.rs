use half::{bf16, f16};
use num_traits::{Float, NumCast};
use proc_macros::kernel;

use crate::array::ArrayElement;

// CPU mirror of the Mode L mega kernel `DeltaNetChunkedMegaApply`. It reproduces
// the per-(head, chunk) math independent of the VT tiling: R = beta*(V - e^g (.)
// (K . S^T)), Vnew = T . R (bf16 T), Y = e^g (.) (Q . S^T) + A . Vnew, and the
// state update S^T <- alpha S^T + (decay (.) K)^T . Vnew. State is f32
// end-to-end; T and A are bf16 device operands.
#[kernel(DeltaNetChunkedMegaApply)]
#[variants(T, f32, f16, bf16)]
#[variants(O, f32, bf16)]
#[variants(VT, 16, 32)]
#[variants(USE_MXU, false, true)]
#[allow(clippy::too_many_arguments)]
pub fn delta_net_chunked_mega_apply<
    T: ArrayElement + Float,
    O: ArrayElement + NumCast,
    const VT: u32,
    const USE_MXU: bool,
>(
    q_norm: *const f32,
    k_norm: *const f32,
    in_proj: *const T,
    qk_scaled: *const f32,
    t_mat: *const bf16,
    g: *const f32,
    beta: *const f32,
    state: *mut f32,
    out: *mut O,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    key_dim: u32,
    value_dim: u32,
    suffix_len: u32,
) {
    let _ = VT;
    let _ = USE_MXU;
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
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;

    let mut r = vec![0.0f32; CHUNK_SIZE * head_v_dim];
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

            // R phase: R = beta*(V - e^g (.) (K . S^T)).
            for local_t in 0..CHUNK_SIZE {
                if local_t >= valid_tokens {
                    for dv in 0..head_v_dim {
                        r[local_t * head_v_dim + dv] = 0.0;
                    }
                    continue;
                }
                let token = token_base + local_t;
                let beta_t = unsafe { *beta.add(token * num_v_heads + hv) };
                let g_scale = unsafe { *g.add(token * num_v_heads + hv) }.exp();
                let k_base = token * key_dim + hk * HEAD_K_DIM;
                let v_base = token * total_proj_dim + 2 * key_dim + hv * head_v_dim;
                for dv in 0..head_v_dim {
                    let state_row = (hv * head_v_dim + dv) * HEAD_K_DIM;
                    let mut correction = 0.0f32;
                    for dk in 0..HEAD_K_DIM {
                        correction += unsafe { *k_norm.add(k_base + dk) } * unsafe { *state.add(state_row + dk) };
                    }
                    let v: f32 = NumCast::from(unsafe { *in_proj.add(v_base + dv) }).unwrap();
                    r[local_t * head_v_dim + dv] = beta_t * (v - g_scale * correction);
                }
            }

            // Vnew phase: Vnew = T . R (T bf16).
            let t_base = chunk_head_base * CHUNK_SIZE * CHUNK_SIZE;
            for i in 0..CHUNK_SIZE {
                for dv in 0..head_v_dim {
                    let mut value = 0.0f32;
                    for k in 0..CHUNK_SIZE {
                        let t_val = unsafe { *t_mat.add(t_base + i * CHUNK_SIZE + k) }.to_f32();
                        value += t_val * r[k * head_v_dim + dv];
                    }
                    vnew[i * head_v_dim + dv] = value;
                }
            }

            // Y phase: Y = e^g (.) (Q . S^T) + A . Vnew -> out.
            for local_t in 0..valid_tokens {
                let token = token_base + local_t;
                let g_scale = unsafe { *g.add(token * num_v_heads + hv) }.exp();
                let q_base = token * key_dim + hk * HEAD_K_DIM;
                let qk_base = chunk_head_base * CHUNK_SIZE * CHUNK_SIZE + local_t * CHUNK_SIZE;
                for dv in 0..head_v_dim {
                    let state_row = (hv * head_v_dim + dv) * HEAD_K_DIM;
                    let mut value = 0.0f32;
                    for dk in 0..HEAD_K_DIM {
                        value += unsafe { *q_norm.add(q_base + dk) } * unsafe { *state.add(state_row + dk) };
                    }
                    value *= g_scale;
                    for j in 0..valid_tokens {
                        let a = unsafe { *qk_scaled.add(qk_base + j) };
                        value += a * vnew[j * head_v_dim + dv];
                    }
                    unsafe { *out.add(token * value_dim + hv * head_v_dim + dv) = NumCast::from(value).unwrap() };
                }
            }

            // Update phase: S^T <- alpha S^T + (decay (.) K)^T . Vnew.
            let g_last = unsafe { *g.add((token_base + valid_tokens - 1) * num_v_heads + hv) };
            let alpha = g_last.exp();
            for dv in 0..head_v_dim {
                let state_row = (hv * head_v_dim + dv) * HEAD_K_DIM;
                for dk in 0..HEAD_K_DIM {
                    let mut value = alpha * unsafe { *state.add(state_row + dk) };
                    for local_t in 0..valid_tokens {
                        let token = token_base + local_t;
                        let k = unsafe { *k_norm.add(token * key_dim + hk * HEAD_K_DIM + dk) };
                        let g_t = unsafe { *g.add(token * num_v_heads + hv) };
                        let decay = (g_last - g_t).exp();
                        value += vnew[local_t * head_v_dim + dv] * k * decay;
                    }
                    unsafe { *state.add(state_row + dk) = value };
                }
            }
        }
    }
}
