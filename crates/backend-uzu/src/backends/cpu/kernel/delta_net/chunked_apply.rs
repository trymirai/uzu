use half::{bf16, f16};
use num_traits::{Float, NumCast};
use proc_macros::kernel;

use crate::array::ArrayElement;

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

#[kernel(DeltaNetChunkedStateA)]
#[variants(HEAD_K_DIM, 128)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
#[variants(RECOMPUTE_G, false, true)]
pub fn delta_net_chunked_state_a<const HEAD_K_DIM: u32, const CHUNK_SIZE: u32, const RECOMPUTE_G: bool>(
    k_norm: *const f32,
    w: *const f32,
    u: *const f32,
    g_or_log_decay: *const f32,
    state: *mut f32,
    h: *mut f32,
    v_new: *mut f32,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    key_dim: u32,
    suffix_len: u32,
    #[allow(unused)] num_dv_groups: u32,
) {
    let head_k_dim = HEAD_K_DIM as usize;
    let chunk_size = CHUNK_SIZE as usize;
    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let head_v_dim = head_v_dim as usize;
    let key_dim = key_dim as usize;
    let suffix_len = suffix_len as usize;
    let num_chunks = suffix_len.div_ceil(chunk_size);
    let groups_per_head = num_v_heads / num_k_heads;

    for hv in 0..num_v_heads {
        let hk = hv / groups_per_head;
        for dv in 0..head_v_dim {
            let state_base = (hv * head_v_dim + dv) * head_k_dim;
            let mut state_row = vec![0.0f32; head_k_dim];
            for dk in 0..head_k_dim {
                state_row[dk] = unsafe { *state.add(state_base + dk) };
            }

            for chunk in 0..num_chunks {
                let token_base = chunk * chunk_size;
                let h_base = ((chunk * num_v_heads + hv) * head_v_dim + dv) * head_k_dim;
                for dk in 0..head_k_dim {
                    unsafe { *h.add(h_base + dk) = state_row[dk] };
                }

                for local_t in 0..chunk_size {
                    let token = token_base + local_t;
                    let out_idx = ((chunk * num_v_heads + hv) * chunk_size + local_t) * head_v_dim + dv;
                    if token >= suffix_len {
                        unsafe { *v_new.add(out_idx) = 0.0 };
                        continue;
                    }
                    let mut correction = 0.0f32;
                    let w_base = ((chunk * num_v_heads + hv) * chunk_size + local_t) * head_k_dim;
                    for dk in 0..head_k_dim {
                        correction += unsafe { *w.add(w_base + dk) } * state_row[dk];
                    }
                    let u_idx = ((chunk * num_v_heads + hv) * chunk_size + local_t) * head_v_dim + dv;
                    unsafe { *v_new.add(out_idx) = *u.add(u_idx) - correction };
                }

                let last_local = (suffix_len - token_base).min(chunk_size).saturating_sub(1);
                let g_last = chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, last_local);
                let g_last_exp = g_last.exp();
                for value in &mut state_row {
                    *value *= g_last_exp;
                }
                for local_t in 0..chunk_size {
                    let token = token_base + local_t;
                    if token >= suffix_len {
                        break;
                    }
                    let scale =
                        unsafe { *v_new.add(((chunk * num_v_heads + hv) * chunk_size + local_t) * head_v_dim + dv) }
                            * (g_last - chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, local_t))
                                .exp();
                    let k_base = token * key_dim + hk * head_k_dim;
                    for dk in 0..head_k_dim {
                        state_row[dk] += unsafe { *k_norm.add(k_base + dk) } * scale;
                    }
                }
            }

            for dk in 0..head_k_dim {
                unsafe { *state.add(state_base + dk) = state_row[dk] };
            }
        }
    }
}

#[kernel(DeltaNetChunkedStateA2Vnew)]
#[variants(WU, f32, bf16)]
#[variants(H, f32, bf16)]
#[variants(HEAD_K_DIM, 128)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
pub fn delta_net_chunked_state_a2_vnew<
    WU: ArrayElement + NumCast,
    H: ArrayElement + NumCast,
    const HEAD_K_DIM: u32,
    const CHUNK_SIZE: u32,
>(
    w: *const WU,
    u: *const WU,
    state: *const f32,
    h: *mut H,
    v_new: *mut f32,
    num_v_heads: u32,
    head_v_dim: u32,
    suffix_len: u32,
    chunk_idx: u32,
) {
    let head_k_dim = HEAD_K_DIM as usize;
    let chunk_size = CHUNK_SIZE as usize;
    let num_v_heads = num_v_heads as usize;
    let head_v_dim = head_v_dim as usize;
    let suffix_len = suffix_len as usize;
    let chunk = chunk_idx as usize;
    let token_base = chunk * chunk_size;
    let valid_tokens = suffix_len.saturating_sub(token_base).min(chunk_size);

    for hv in 0..num_v_heads {
        for dv in 0..head_v_dim {
            let state_base = (hv * head_v_dim + dv) * head_k_dim;
            let h_base = ((chunk * num_v_heads + hv) * head_v_dim + dv) * head_k_dim;
            for dk in 0..head_k_dim {
                unsafe { *h.add(h_base + dk) = NumCast::from(*state.add(state_base + dk)).unwrap() };
            }
            for local_t in 0..chunk_size {
                let out_idx = ((chunk * num_v_heads + hv) * chunk_size + local_t) * head_v_dim + dv;
                if local_t >= valid_tokens {
                    unsafe { *v_new.add(out_idx) = 0.0 };
                    continue;
                }
                let mut correction = 0.0f32;
                let w_base = ((chunk * num_v_heads + hv) * chunk_size + local_t) * head_k_dim;
                for dk in 0..head_k_dim {
                    let w_value: f32 = NumCast::from(unsafe { *w.add(w_base + dk) }).unwrap();
                    correction += w_value * unsafe { *state.add(state_base + dk) };
                }
                let u_idx = ((chunk * num_v_heads + hv) * chunk_size + local_t) * head_v_dim + dv;
                let u_value: f32 = NumCast::from(unsafe { *u.add(u_idx) }).unwrap();
                unsafe { *v_new.add(out_idx) = u_value - correction };
            }
        }
    }
}

#[kernel(DeltaNetChunkedStateA2Update)]
#[variants(HEAD_K_DIM, 128)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
#[variants(RECOMPUTE_G, false, true)]
pub fn delta_net_chunked_state_a2_update<const HEAD_K_DIM: u32, const CHUNK_SIZE: u32, const RECOMPUTE_G: bool>(
    k_norm: *const f32,
    g_or_log_decay: *const f32,
    v_new: *const f32,
    state: *mut f32,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    key_dim: u32,
    suffix_len: u32,
    chunk_idx: u32,
) {
    let head_k_dim = HEAD_K_DIM as usize;
    let chunk_size = CHUNK_SIZE as usize;
    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let head_v_dim = head_v_dim as usize;
    let key_dim = key_dim as usize;
    let suffix_len = suffix_len as usize;
    let chunk = chunk_idx as usize;
    let token_base = chunk * chunk_size;
    let valid_tokens = suffix_len.saturating_sub(token_base).min(chunk_size);
    if valid_tokens == 0 {
        return;
    }
    let groups_per_head = num_v_heads / num_k_heads;

    for hv in 0..num_v_heads {
        let hk = hv / groups_per_head;
        let g_last = chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, valid_tokens - 1);
        let alpha = g_last.exp();
        for dv in 0..head_v_dim {
            let state_base = (hv * head_v_dim + dv) * head_k_dim;
            for dk in 0..head_k_dim {
                let mut value = unsafe { *state.add(state_base + dk) } * alpha;
                for local_t in 0..valid_tokens {
                    let token = token_base + local_t;
                    let decay =
                        (g_last - chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, local_t)).exp();
                    let v =
                        unsafe { *v_new.add(((chunk * num_v_heads + hv) * chunk_size + local_t) * head_v_dim + dv) };
                    let k = unsafe { *k_norm.add(token * key_dim + hk * head_k_dim + dk) };
                    value += v * decay * k;
                }
                unsafe { *state.add(state_base + dk) = value };
            }
        }
    }
}

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

#[kernel(DeltaNetChunkedStateA2UpdateDecayScale)]
#[variants(HEAD_K_DIM, 128)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
pub fn delta_net_chunked_state_a2_update_decay_scale<const HEAD_K_DIM: u32, const CHUNK_SIZE: u32>(
    k_norm: *const f32,
    g: *const f32,
    decay_scale: *const f32,
    v_new: *const f32,
    state: *mut f32,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    key_dim: u32,
    suffix_len: u32,
    chunk_idx: u32,
) {
    let head_k_dim = HEAD_K_DIM as usize;
    let chunk_size = CHUNK_SIZE as usize;
    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let head_v_dim = head_v_dim as usize;
    let key_dim = key_dim as usize;
    let suffix_len = suffix_len as usize;
    let chunk = chunk_idx as usize;
    let token_base = chunk * chunk_size;
    let valid_tokens = suffix_len.saturating_sub(token_base).min(chunk_size);
    if valid_tokens == 0 {
        return;
    }
    let groups_per_head = num_v_heads / num_k_heads;

    for hv in 0..num_v_heads {
        let hk = hv / groups_per_head;
        let g_last = unsafe { *g.add((token_base + valid_tokens - 1) * num_v_heads + hv) };
        let alpha = g_last.exp();
        for dv in 0..head_v_dim {
            let state_base = (hv * head_v_dim + dv) * head_k_dim;
            for dk in 0..head_k_dim {
                let mut value = unsafe { *state.add(state_base + dk) } * alpha;
                for local_t in 0..valid_tokens {
                    let v =
                        unsafe { *v_new.add(((chunk * num_v_heads + hv) * chunk_size + local_t) * head_v_dim + dv) };
                    let k = unsafe { *k_norm.add((token_base + local_t) * key_dim + hk * head_k_dim + dk) };
                    let decay = unsafe { *decay_scale.add((chunk * num_v_heads + hv) * chunk_size + local_t) };
                    value += v * k * decay;
                }
                unsafe { *state.add(state_base + dk) = value };
            }
        }
    }
}

#[kernel(DeltaNetChunkedOutputA)]
#[variants(T, f32, f16, bf16)]
#[variants(H, f32, bf16)]
#[variants(HEAD_K_DIM, 128)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
#[variants(RECOMPUTE_G, false, true)]
pub fn delta_net_chunked_output_a<
    T: ArrayElement + Float,
    H: ArrayElement + NumCast,
    const HEAD_K_DIM: u32,
    const CHUNK_SIZE: u32,
    const RECOMPUTE_G: bool,
>(
    q_norm: *const f32,
    qk: *const f32,
    g_or_log_decay: *const f32,
    h: *const H,
    v_new: *const f32,
    out: *mut T,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    key_dim: u32,
    value_dim: u32,
    suffix_len: u32,
    #[allow(unused)] num_dv_groups: u32,
) {
    let head_k_dim = HEAD_K_DIM as usize;
    let chunk_size = CHUNK_SIZE as usize;
    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let head_v_dim = head_v_dim as usize;
    let key_dim = key_dim as usize;
    let value_dim = value_dim as usize;
    let suffix_len = suffix_len as usize;
    let num_chunks = suffix_len.div_ceil(chunk_size);
    let groups_per_head = num_v_heads / num_k_heads;

    for chunk in 0..num_chunks {
        let token_base = chunk * chunk_size;
        for hv in 0..num_v_heads {
            let hk = hv / groups_per_head;
            for local_t in 0..chunk_size {
                let token = token_base + local_t;
                if token >= suffix_len {
                    continue;
                }
                let g_row = chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, local_t);
                let q_base = token * key_dim + hk * head_k_dim;
                let qk_base = (chunk * num_k_heads + hk) * chunk_size * chunk_size + local_t * chunk_size;

                for dv in 0..head_v_dim {
                    let h_base = ((chunk * num_v_heads + hv) * head_v_dim + dv) * head_k_dim;
                    let mut value = 0.0f32;
                    for dk in 0..head_k_dim {
                        let h_value: f32 = NumCast::from(unsafe { *h.add(h_base + dk) }).unwrap();
                        value += unsafe { *q_norm.add(q_base + dk) } * h_value;
                    }
                    value *= g_row.exp();

                    for local_j in 0..=local_t {
                        let source_token = token_base + local_j;
                        if source_token >= suffix_len {
                            break;
                        }
                        let qk_value = unsafe { *qk.add(qk_base + local_j) };
                        let source_g = chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, local_j);
                        let v_value = unsafe {
                            *v_new.add(((chunk * num_v_heads + hv) * chunk_size + local_j) * head_v_dim + dv)
                        };
                        value += qk_value * (g_row - source_g).exp() * v_value;
                    }

                    unsafe { *out.add(token * value_dim + hv * head_v_dim + dv) = T::from(value).unwrap() };
                }
            }
        }
    }
}

#[kernel(DeltaNetChunkedOutputAScaledQk)]
#[variants(T, f32, f16, bf16)]
#[variants(H, f32, bf16)]
#[variants(HEAD_K_DIM, 128)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
pub fn delta_net_chunked_output_a_scaled_qk<
    T: ArrayElement + Float,
    H: ArrayElement + NumCast,
    const HEAD_K_DIM: u32,
    const CHUNK_SIZE: u32,
>(
    q_norm: *const f32,
    qk_scaled: *const f32,
    g: *const f32,
    h: *const H,
    v_new: *const f32,
    out: *mut T,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    key_dim: u32,
    value_dim: u32,
    suffix_len: u32,
    #[allow(unused)] num_dv_groups: u32,
) {
    let head_k_dim = HEAD_K_DIM as usize;
    let chunk_size = CHUNK_SIZE as usize;
    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let head_v_dim = head_v_dim as usize;
    let key_dim = key_dim as usize;
    let value_dim = value_dim as usize;
    let suffix_len = suffix_len as usize;
    let num_chunks = suffix_len.div_ceil(chunk_size);
    let groups_per_head = num_v_heads / num_k_heads;

    for chunk in 0..num_chunks {
        let token_base = chunk * chunk_size;
        for hv in 0..num_v_heads {
            let hk = hv / groups_per_head;
            for local_t in 0..chunk_size {
                let token = token_base + local_t;
                if token >= suffix_len {
                    continue;
                }
                let g_row = unsafe { *g.add(token * num_v_heads + hv) };
                let q_base = token * key_dim + hk * head_k_dim;
                let qk_base = (chunk * num_v_heads + hv) * chunk_size * chunk_size + local_t * chunk_size;

                for dv in 0..head_v_dim {
                    let h_base = ((chunk * num_v_heads + hv) * head_v_dim + dv) * head_k_dim;
                    let mut value = 0.0f32;
                    for dk in 0..head_k_dim {
                        let h_value: f32 = NumCast::from(unsafe { *h.add(h_base + dk) }).unwrap();
                        value += unsafe { *q_norm.add(q_base + dk) } * h_value;
                    }
                    value *= g_row.exp();

                    for local_j in 0..=local_t {
                        let source_token = token_base + local_j;
                        if source_token >= suffix_len {
                            break;
                        }
                        let qk_value = unsafe { *qk_scaled.add(qk_base + local_j) };
                        let v_value = unsafe {
                            *v_new.add(((chunk * num_v_heads + hv) * chunk_size + local_j) * head_v_dim + dv)
                        };
                        value += qk_value * v_value;
                    }

                    unsafe { *out.add(token * value_dim + hv * head_v_dim + dv) = T::from(value).unwrap() };
                }
            }
        }
    }
}

#[kernel(DeltaNetChunkedStateC)]
#[variants(HEAD_K_DIM, 128)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
#[variants(RECOMPUTE_G, false, true)]
pub fn delta_net_chunked_state_c<const HEAD_K_DIM: u32, const CHUNK_SIZE: u32, const RECOMPUTE_G: bool>(
    k_norm: *const f32,
    w: *const f32,
    u: *const f32,
    g_or_log_decay: *const f32,
    state: *mut f32,
    h: *mut f32,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    key_dim: u32,
    suffix_len: u32,
    #[allow(unused)] num_dv_groups: u32,
) {
    let head_k_dim = HEAD_K_DIM as usize;
    let chunk_size = CHUNK_SIZE as usize;
    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let head_v_dim = head_v_dim as usize;
    let key_dim = key_dim as usize;
    let suffix_len = suffix_len as usize;
    let num_chunks = suffix_len.div_ceil(chunk_size);
    let groups_per_head = num_v_heads / num_k_heads;

    for hv in 0..num_v_heads {
        let hk = hv / groups_per_head;
        for dv in 0..head_v_dim {
            let state_base = (hv * head_v_dim + dv) * head_k_dim;
            let mut state_row = vec![0.0f32; head_k_dim];
            for dk in 0..head_k_dim {
                state_row[dk] = unsafe { *state.add(state_base + dk) };
            }

            for chunk in 0..num_chunks {
                let token_base = chunk * chunk_size;
                let h_base = ((chunk * num_v_heads + hv) * head_v_dim + dv) * head_k_dim;
                for dk in 0..head_k_dim {
                    unsafe { *h.add(h_base + dk) = state_row[dk] };
                }

                let last_local = (suffix_len - token_base).min(chunk_size).saturating_sub(1);
                let g_last = chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, last_local);
                let g_last_exp = g_last.exp();
                let mut next_state = state_row.iter().map(|value| value * g_last_exp).collect::<Vec<_>>();

                for local_t in 0..chunk_size {
                    let token = token_base + local_t;
                    if token >= suffix_len {
                        break;
                    }
                    let mut correction = 0.0f32;
                    let w_base = ((chunk * num_v_heads + hv) * chunk_size + local_t) * head_k_dim;
                    for dk in 0..head_k_dim {
                        correction += unsafe { *w.add(w_base + dk) } * state_row[dk];
                    }
                    let u_idx = ((chunk * num_v_heads + hv) * chunk_size + local_t) * head_v_dim + dv;
                    let v_value = unsafe { *u.add(u_idx) } - correction;
                    let scale = v_value
                        * (g_last - chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, local_t))
                            .exp();
                    let k_base = token * key_dim + hk * head_k_dim;
                    for dk in 0..head_k_dim {
                        next_state[dk] += unsafe { *k_norm.add(k_base + dk) } * scale;
                    }
                }
                state_row = next_state;
            }

            for dk in 0..head_k_dim {
                unsafe { *state.add(state_base + dk) = state_row[dk] };
            }
        }
    }
}

#[kernel(DeltaNetChunkedOutputC)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_K_DIM, 128)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
#[variants(RECOMPUTE_G, false, true)]
pub fn delta_net_chunked_output_c<
    T: ArrayElement + Float,
    const HEAD_K_DIM: u32,
    const CHUNK_SIZE: u32,
    const RECOMPUTE_G: bool,
>(
    q_norm: *const f32,
    qk: *const f32,
    g_or_log_decay: *const f32,
    h: *const f32,
    w: *const f32,
    u: *const f32,
    out: *mut T,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    key_dim: u32,
    value_dim: u32,
    suffix_len: u32,
    #[allow(unused)] num_dv_groups: u32,
) {
    let head_k_dim = HEAD_K_DIM as usize;
    let chunk_size = CHUNK_SIZE as usize;
    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let head_v_dim = head_v_dim as usize;
    let key_dim = key_dim as usize;
    let value_dim = value_dim as usize;
    let suffix_len = suffix_len as usize;
    let num_chunks = suffix_len.div_ceil(chunk_size);
    let groups_per_head = num_v_heads / num_k_heads;

    for chunk in 0..num_chunks {
        let token_base = chunk * chunk_size;
        for hv in 0..num_v_heads {
            let hk = hv / groups_per_head;
            for local_t in 0..chunk_size {
                let token = token_base + local_t;
                if token >= suffix_len {
                    continue;
                }
                let g_row = chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, local_t);
                let q_base = token * key_dim + hk * head_k_dim;
                let qk_base = (chunk * num_k_heads + hk) * chunk_size * chunk_size + local_t * chunk_size;

                for dv in 0..head_v_dim {
                    let h_base = ((chunk * num_v_heads + hv) * head_v_dim + dv) * head_k_dim;
                    let mut value = 0.0f32;
                    for dk in 0..head_k_dim {
                        value += unsafe { *q_norm.add(q_base + dk) } * unsafe { *h.add(h_base + dk) };
                    }
                    value *= g_row.exp();

                    for local_j in 0..=local_t {
                        let source_token = token_base + local_j;
                        if source_token >= suffix_len {
                            break;
                        }
                        let mut correction = 0.0f32;
                        let w_base = ((chunk * num_v_heads + hv) * chunk_size + local_j) * head_k_dim;
                        for dk in 0..head_k_dim {
                            correction += unsafe { *w.add(w_base + dk) } * unsafe { *h.add(h_base + dk) };
                        }
                        let u_idx = ((chunk * num_v_heads + hv) * chunk_size + local_j) * head_v_dim + dv;
                        let v_value = unsafe { *u.add(u_idx) } - correction;
                        let qk_value = unsafe { *qk.add(qk_base + local_j) };
                        let source_g = chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, local_j);
                        value += qk_value * (g_row - source_g).exp() * v_value;
                    }

                    unsafe { *out.add(token * value_dim + hv * head_v_dim + dv) = T::from(value).unwrap() };
                }
            }
        }
    }
}

#[kernel(DeltaNetChunkedApplyB)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_K_DIM, 128)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
#[variants(RECOMPUTE_G, false, true)]
pub fn delta_net_chunked_apply_b<
    T: ArrayElement + Float,
    const HEAD_K_DIM: u32,
    const CHUNK_SIZE: u32,
    const RECOMPUTE_G: bool,
>(
    q_norm: *const f32,
    k_norm: *const f32,
    qk: *const f32,
    g_or_log_decay: *const f32,
    w: *const f32,
    u: *const f32,
    state: *mut f32,
    out: *mut T,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    key_dim: u32,
    value_dim: u32,
    suffix_len: u32,
    #[allow(unused)] num_dv_groups: u32,
) {
    let head_k_dim = HEAD_K_DIM as usize;
    let chunk_size = CHUNK_SIZE as usize;
    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let head_v_dim = head_v_dim as usize;
    let key_dim = key_dim as usize;
    let value_dim = value_dim as usize;
    let suffix_len = suffix_len as usize;
    let num_chunks = suffix_len.div_ceil(chunk_size);
    let groups_per_head = num_v_heads / num_k_heads;

    for hv in 0..num_v_heads {
        let hk = hv / groups_per_head;
        for dv in 0..head_v_dim {
            let state_base = (hv * head_v_dim + dv) * head_k_dim;
            let mut state_row = vec![0.0f32; head_k_dim];
            for dk in 0..head_k_dim {
                state_row[dk] = unsafe { *state.add(state_base + dk) };
            }

            for chunk in 0..num_chunks {
                let token_base = chunk * chunk_size;
                for local_t in 0..chunk_size {
                    let token = token_base + local_t;
                    if token >= suffix_len {
                        continue;
                    }
                    let g_row = chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, local_t);
                    let q_base = token * key_dim + hk * head_k_dim;
                    let qk_base = (chunk * num_k_heads + hk) * chunk_size * chunk_size + local_t * chunk_size;

                    let mut value = 0.0f32;
                    for dk in 0..head_k_dim {
                        value += unsafe { *q_norm.add(q_base + dk) } * state_row[dk];
                    }
                    value *= g_row.exp();

                    for local_j in 0..=local_t {
                        let source_token = token_base + local_j;
                        if source_token >= suffix_len {
                            break;
                        }
                        let mut correction = 0.0f32;
                        let w_base = ((chunk * num_v_heads + hv) * chunk_size + local_j) * head_k_dim;
                        for dk in 0..head_k_dim {
                            correction += unsafe { *w.add(w_base + dk) } * state_row[dk];
                        }
                        let u_idx = ((chunk * num_v_heads + hv) * chunk_size + local_j) * head_v_dim + dv;
                        let v_value = unsafe { *u.add(u_idx) } - correction;
                        let qk_value = unsafe { *qk.add(qk_base + local_j) };
                        let source_g = chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, local_j);
                        value += qk_value * (g_row - source_g).exp() * v_value;
                    }

                    unsafe { *out.add(token * value_dim + hv * head_v_dim + dv) = T::from(value).unwrap() };
                }

                let last_local = (suffix_len - token_base).min(chunk_size).saturating_sub(1);
                let g_last = chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, last_local);
                let g_last_exp = g_last.exp();
                let mut next_state = state_row.iter().map(|value| value * g_last_exp).collect::<Vec<_>>();

                for local_t in 0..chunk_size {
                    let token = token_base + local_t;
                    if token >= suffix_len {
                        break;
                    }
                    let mut correction = 0.0f32;
                    let w_base = ((chunk * num_v_heads + hv) * chunk_size + local_t) * head_k_dim;
                    for dk in 0..head_k_dim {
                        correction += unsafe { *w.add(w_base + dk) } * state_row[dk];
                    }
                    let u_idx = ((chunk * num_v_heads + hv) * chunk_size + local_t) * head_v_dim + dv;
                    let v_value = unsafe { *u.add(u_idx) } - correction;
                    let scale = v_value
                        * (g_last - chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, local_t))
                            .exp();
                    let k_base = token * key_dim + hk * head_k_dim;
                    for dk in 0..head_k_dim {
                        next_state[dk] += unsafe { *k_norm.add(k_base + dk) } * scale;
                    }
                }
                state_row = next_state;
            }

            for dk in 0..head_k_dim {
                unsafe { *state.add(state_base + dk) = state_row[dk] };
            }
        }
    }
}
