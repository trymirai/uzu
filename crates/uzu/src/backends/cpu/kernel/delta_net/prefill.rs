use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(DeltaNetPrefill)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_K_DIM, 128)]
pub fn delta_net_prefill<T: ArrayElement + Float, const HEAD_K_DIM: u32>(
    q_norm: *const f32,
    k_norm: *const f32,
    beta_buf: *const f32,
    decay_buf: *const f32,
    in_proj: *const T,
    state: *mut T,
    out: *mut T,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    key_dim: u32,
    value_dim: u32,
    suffix_len: u32,
    num_dv_groups: u32,
) {
    let _ = num_dv_groups;
    let state_ptr = state as *const T;
    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let head_k_dim = HEAD_K_DIM as usize;
    let head_v_dim = head_v_dim as usize;
    let key_dim = key_dim as usize;
    let value_dim = value_dim as usize;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let suffix_len = suffix_len as usize;
    let groups_per_head = num_v_heads / num_k_heads;

    for hv in 0..num_v_heads {
        let hk = hv / groups_per_head;

        for token in 0..suffix_len {
            let qk_off = token * key_dim + hk * head_k_dim;
            let decay = unsafe { *decay_buf.add(token * num_v_heads + hv) };
            let beta = unsafe { *beta_buf.add(token * num_v_heads + hv) };

            for i in 0..head_v_dim {
                // State layout: [Hv, Dv, Dk]
                let state_off = (hv * head_v_dim + i) * head_k_dim;

                let mut kv_mem = 0.0f32;
                for j in 0..head_k_dim {
                    let s = unsafe { (*state_ptr.add(state_off + j)).to_f32().unwrap() };
                    kv_mem += (decay * s) * unsafe { *k_norm.add(qk_off + j) };
                }

                let v_val = unsafe {
                    (*in_proj.add(token * total_proj_dim + 2 * key_dim + hv * head_v_dim + i)).to_f32().unwrap()
                };
                let delta = beta * (v_val - kv_mem);

                let mut o_val = 0.0f32;
                for j in 0..head_k_dim {
                    let s = unsafe { (*state_ptr.add(state_off + j)).to_f32().unwrap() };
                    let k_j = unsafe { *k_norm.add(qk_off + j) };
                    let new_s = decay * s + k_j * delta;
                    unsafe { *state.add(state_off + j) = T::from(new_s).unwrap() };
                    o_val += new_s * unsafe { *q_norm.add(qk_off + j) };
                }

                unsafe { *out.add(token * value_dim + hv * head_v_dim + i) = T::from(o_val).unwrap() };
            }
        }
    }
}
