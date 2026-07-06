use half::{bf16, f16};
use num_traits::{Float, NumCast};
use proc_macros::kernel;

use crate::array::ArrayElement;

// CPU mirror of the Mode S mega kernel `DeltaNetChunkedMegaApplyInline`. It
// reproduces the inline precompute (chunk-local g prefix, gram, block 16x16
// solve -> dense inverse T) and the per-(head, chunk) scan of the Mode L mega
// kernel, but computing A (qk_scaled) and T itself from q_norm/k_norm/log_decay/
// beta instead of reading them from device. Scratch matrices (A_strict, Dinv, T,
// QK) are bf16 -- matching the Metal kernel's threadgroup dtypes -- while state
// is f32 end-to-end. T is built into a separate buffer here (the Metal kernel
// does it in place); the numerics are identical.
#[kernel(DeltaNetChunkedMegaApplyInline)]
#[variants(T, f32, f16, bf16)]
#[variants(O, f32, bf16)]
#[variants(VT, 16)]
#[allow(clippy::too_many_arguments)]
pub fn delta_net_chunked_mega_apply_inline<T: ArrayElement + Float, O: ArrayElement + NumCast, const VT: u32>(
    q_norm: *const f32,
    k_norm: *const f32,
    in_proj: *const T,
    log_decay: *const f32,
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
    const HEAD_K_DIM: usize = 128;
    const CHUNK_SIZE: usize = 64;
    const BLOCK: usize = 16;
    const NUM_BLOCKS: usize = CHUNK_SIZE / BLOCK;

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

    let bf = |x: f32| bf16::from_f32(x).to_f32();

    let mut gtile = vec![0.0f32; CHUNK_SIZE];
    let mut amat = vec![0.0f32; CHUNK_SIZE * CHUNK_SIZE]; // A_strict (bf16-rounded)
    let mut qk = vec![0.0f32; CHUNK_SIZE * CHUNK_SIZE]; // QK (bf16-rounded)
    let mut dinv = vec![0.0f32; NUM_BLOCKS * BLOCK * BLOCK]; // diagonal-block inverses (bf16)
    let mut t = vec![0.0f32; CHUNK_SIZE * CHUNK_SIZE]; // dense inverse T (bf16-rounded)
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

            // gtile: chunk-local cumsum of log_decay.
            for local in 0..CHUNK_SIZE {
                let mut acc = 0.0f32;
                if local < valid_tokens {
                    for i in 0..=local {
                        acc += unsafe { *log_decay.add((token_base + i) * num_v_heads + hv) };
                    }
                }
                gtile[local] = acc;
            }

            // gram -> A_strict (bf16) and QK (bf16).
            for row in 0..CHUNK_SIZE {
                for col in 0..CHUNK_SIZE {
                    let (kk, qkv) = if row < valid_tokens && col < valid_tokens {
                        let k_row = (token_base + row) * key_dim + hk * HEAD_K_DIM;
                        let k_col = (token_base + col) * key_dim + hk * HEAD_K_DIM;
                        let mut kk = 0.0f32;
                        let mut qkv = 0.0f32;
                        for d in 0..HEAD_K_DIM {
                            let kc = unsafe { *k_norm.add(k_col + d) };
                            kk += unsafe { *k_norm.add(k_row + d) } * kc;
                            qkv += unsafe { *q_norm.add(k_row + d) } * kc;
                        }
                        (kk, qkv)
                    } else {
                        (0.0, 0.0)
                    };
                    amat[row * CHUNK_SIZE + col] = if row < valid_tokens && col < valid_tokens && col < row {
                        let beta_r = unsafe { *beta.add((token_base + row) * num_v_heads + hv) };
                        bf(beta_r * (gtile[row] - gtile[col]).exp() * kk)
                    } else {
                        0.0
                    };
                    qk[row * CHUNK_SIZE + col] = if row < valid_tokens && col < valid_tokens && col <= row {
                        bf((gtile[row] - gtile[col]).exp() * qkv)
                    } else {
                        0.0
                    };
                }
            }

            // solve step 1: diagonal-block inverses (unit lower triangular).
            for b in 0..NUM_BLOCKS {
                for inv_col in 0..BLOCK {
                    let mut inverse_col = [0.0f32; BLOCK];
                    inverse_col[inv_col] = 1.0;
                    for inv_row in 0..BLOCK {
                        if inv_row > inv_col {
                            let mut acc = 0.0f32;
                            for prev in 0..inv_row {
                                let a = amat[(b * BLOCK + inv_row) * CHUNK_SIZE + (b * BLOCK + prev)];
                                acc += a * inverse_col[prev];
                            }
                            inverse_col[inv_row] = -acc;
                        }
                    }
                    for inv_row in 0..BLOCK {
                        dinv[b * BLOCK * BLOCK + inv_row * BLOCK + inv_col] = bf(inverse_col[inv_row]);
                    }
                }
            }

            // solve step 2: block forward substitution -> dense T (bf16).
            for i in 0..NUM_BLOCKS {
                let mut acc = vec![0.0f32; BLOCK * CHUNK_SIZE];
                for rl in 0..BLOCK {
                    let gr = i * BLOCK + rl;
                    acc[rl * CHUNK_SIZE + gr] = 1.0;
                    for j in 0..i {
                        for jl in 0..BLOCK {
                            let a = amat[(i * BLOCK + rl) * CHUNK_SIZE + (j * BLOCK + jl)];
                            let t_prev = j * BLOCK + jl;
                            for c in 0..CHUNK_SIZE {
                                acc[rl * CHUNK_SIZE + c] -= a * t[t_prev * CHUNK_SIZE + c];
                            }
                        }
                    }
                }
                for rl in 0..BLOCK {
                    for c in 0..CHUNK_SIZE {
                        let mut value = 0.0f32;
                        for src in 0..BLOCK {
                            value += dinv[i * BLOCK * BLOCK + rl * BLOCK + src] * acc[src * CHUNK_SIZE + c];
                        }
                        t[(i * BLOCK + rl) * CHUNK_SIZE + c] = bf(value);
                    }
                }
            }

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
                let g_scale = gtile[local_t].exp();
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
            for i in 0..CHUNK_SIZE {
                for dv in 0..head_v_dim {
                    let mut value = 0.0f32;
                    for k in 0..CHUNK_SIZE {
                        value += t[i * CHUNK_SIZE + k] * r[k * head_v_dim + dv];
                    }
                    vnew[i * head_v_dim + dv] = value;
                }
            }

            // Y phase: Y = e^g (.) (Q . S^T) + A . Vnew -> out.
            for local_t in 0..valid_tokens {
                let token = token_base + local_t;
                let g_scale = gtile[local_t].exp();
                let q_base = token * key_dim + hk * HEAD_K_DIM;
                for dv in 0..head_v_dim {
                    let state_row = (hv * head_v_dim + dv) * HEAD_K_DIM;
                    let mut value = 0.0f32;
                    for dk in 0..HEAD_K_DIM {
                        value += unsafe { *q_norm.add(q_base + dk) } * unsafe { *state.add(state_row + dk) };
                    }
                    value *= g_scale;
                    for j in 0..valid_tokens {
                        value += qk[local_t * CHUNK_SIZE + j] * vnew[j * head_v_dim + dv];
                    }
                    unsafe { *out.add(token * value_dim + hv * head_v_dim + dv) = NumCast::from(value).unwrap() };
                }
            }

            // Update phase: S^T <- alpha S^T + (decay (.) K)^T . Vnew.
            let g_last = gtile[valid_tokens - 1];
            let alpha = g_last.exp();
            for dv in 0..head_v_dim {
                let state_row = (hv * head_v_dim + dv) * HEAD_K_DIM;
                for dk in 0..HEAD_K_DIM {
                    let mut value = alpha * unsafe { *state.add(state_row + dk) };
                    for local_t in 0..valid_tokens {
                        let token = token_base + local_t;
                        let k = unsafe { *k_norm.add(token * key_dim + hk * HEAD_K_DIM + dk) };
                        let decay = (g_last - gtile[local_t]).exp();
                        value += vnew[local_t * head_v_dim + dv] * k * decay;
                    }
                    unsafe { *state.add(state_row + dk) = value };
                }
            }
        }
    }
}
