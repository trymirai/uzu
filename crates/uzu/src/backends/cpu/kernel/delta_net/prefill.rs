use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(DeltaNetPrefill)]
#[variants(T, f32, f16, bf16)]
pub fn delta_net_prefill<T: ArrayElement + Float>(
    q_norm: *const f32,
    k_norm: *const f32,
    beta_buf: *const f32,
    decay_buf: *const f32,
    in_proj: *const T,
    state: *mut T,
    out: *mut T,
    num_v_heads: u32,
    num_k_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    key_dim: u32,
    value_dim: u32,
    suffix_len: u32,
    #[allow(unused)] num_dv_groups: u32,
) {
    let state_ptr = state as *const T;
    let nv = num_v_heads as usize;
    let nk = num_k_heads as usize;
    let dk = head_k_dim as usize;
    let dv = head_v_dim as usize;
    let kd = key_dim as usize;
    let vd = value_dim as usize;
    let conv_dim = 2 * kd + vd;
    let total_proj_dim = conv_dim + vd + nv + nv;
    let sl = suffix_len as usize;
    let groups_per_head = nv / nk;

    for hv in 0..nv {
        let hk = hv / groups_per_head;

        for t in 0..sl {
            let qk_off = t * kd + hk * dk;
            let decay = unsafe { *decay_buf.add(t * nv + hv) };
            let beta = unsafe { *beta_buf.add(t * nv + hv) };

            for dv_idx in 0..dv {
                // State layout: [Hv, Dv, Dk]
                let state_off = (hv * dv + dv_idx) * dk;

                let mut kv_mem = 0.0f32;
                for j in 0..dk {
                    let s = unsafe { (*state_ptr.add(state_off + j)).to_f32().unwrap() };
                    let decayed = decay * s;
                    unsafe { *state.add(state_off + j) = T::from(decayed).unwrap() };
                    kv_mem += decayed * unsafe { *k_norm.add(qk_off + j) };
                }

                let v_val = unsafe { (*in_proj.add(t * total_proj_dim + 2 * kd + hv * dv + dv_idx)).to_f32().unwrap() };
                let delta = beta * (v_val - kv_mem);

                let mut o_val = 0.0f32;
                for j in 0..dk {
                    let s = unsafe { (*state_ptr.add(state_off + j)).to_f32().unwrap() };
                    let k_j = unsafe { *k_norm.add(qk_off + j) };
                    let new_s = s + k_j * delta;
                    unsafe { *state.add(state_off + j) = T::from(new_s).unwrap() };
                    o_val += new_s * unsafe { *q_norm.add(qk_off + j) };
                }

                unsafe { *out.add(t * vd + hv * dv + dv_idx) = T::from(o_val).unwrap() };
            }
        }
    }
}
