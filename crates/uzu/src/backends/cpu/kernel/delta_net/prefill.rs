use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

// CPU reference for tiled prefill variant. Same algorithm as DeltaNetPrefill;
#[kernel(DeltaNetPrefill)]
#[variants(T, f32, f16, bf16)]
pub fn delta_net_prefill<T: ArrayElement + Float>(
    in_proj: *const T,
    a_log: *const T,
    dt_bias: *const T,
    state: *mut T,
    out: *mut T,
    num_v_heads: u32,
    num_k_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    key_dim: u32,
    value_dim: u32,
    suffix_len: u32,
    #[allow(unused)] num_v_tiles: u32,
) {
    let state_ptr = state as *const T;

    let nv = num_v_heads as usize;
    let nk = num_k_heads as usize;
    debug_assert!(nv % nk == 0, "num_v_heads must be a multiple of num_k_heads");
    let dk = head_k_dim as usize;
    let dv = head_v_dim as usize;
    let kd = key_dim as usize;
    let vd = value_dim as usize;
    let conv_dim = 2 * kd + vd;
    let total_proj_dim = conv_dim + vd + nv + nv;
    let sl = suffix_len as usize;

    for hv in 0..nv {
        let hk = hv / (nv / nk);
        let state_head_offset = hv * dk * dv;
        let a_log_val = unsafe { (*a_log.add(hv)).to_f32().unwrap() };
        let dt_bias_val = unsafe { (*dt_bias.add(hv)).to_f32().unwrap() };

        let mut q = vec![0.0f32; dk];
        let mut k = vec![0.0f32; dk];

        for t in 0..sl {
            let token_offset = t * total_proj_dim;

            let q_offset = token_offset + hk * dk;
            let k_offset = token_offset + kd + hk * dk;

            for j in 0..dk {
                q[j] = unsafe { (*in_proj.add(q_offset + j)).to_f32().unwrap() };
                k[j] = unsafe { (*in_proj.add(k_offset + j)).to_f32().unwrap() };
            }

            let q_norm_sq: f32 = q.iter().map(|x| x * x).sum();
            let k_norm_sq: f32 = k.iter().map(|x| x * x).sum();
            let q_inv_norm = 1.0 / (q_norm_sq + 1e-6).sqrt();
            let k_inv_norm = 1.0 / (k_norm_sq + 1e-6).sqrt();
            for j in 0..dk {
                q[j] *= q_inv_norm;
                k[j] *= k_inv_norm;
            }

            let q_scale = 1.0 / (dk as f32).sqrt();
            for j in 0..dk {
                q[j] *= q_scale;
            }

            let beta_raw = unsafe { (*in_proj.add(token_offset + conv_dim + vd + hv)).to_f32().unwrap() };
            let beta = 1.0 / (1.0 + (-beta_raw).exp());

            let a_raw = unsafe { (*in_proj.add(token_offset + conv_dim + vd + nv + hv)).to_f32().unwrap() };
            let sp_input = a_raw + dt_bias_val;
            let sp = if sp_input > 20.0 {
                sp_input
            } else {
                (1.0 + sp_input.exp()).ln()
            };
            let g = -a_log_val.exp() * sp;
            let decay = g.exp();

            let kq_dot: f32 = k.iter().zip(q.iter()).map(|(ki, qi)| ki * qi).sum();

            for i in 0..dv {
                let v_i = unsafe { (*in_proj.add(token_offset + 2 * kd + hv * dv + i)).to_f32().unwrap() };

                let mut sq_acc = 0.0f32;
                let mut sk_acc = 0.0f32;
                for j in 0..dk {
                    let s = unsafe { (*state_ptr.add(state_head_offset + j * dv + i)).to_f32().unwrap() };
                    sq_acc += s * q[j];
                    sk_acc += s * k[j];
                }

                let retrieved_i = decay * sk_acc;
                let delta_i = beta * (v_i - retrieved_i);
                let o_i = decay * sq_acc + delta_i * kq_dot;

                unsafe {
                    *out.add(t * vd + hv * dv + i) = T::from(o_i).unwrap();
                }

                for j in 0..dk {
                    let s = unsafe { (*state_ptr.add(state_head_offset + j * dv + i)).to_f32().unwrap() };
                    unsafe {
                        *state.add(state_head_offset + j * dv + i) = T::from(decay * s + k[j] * delta_i).unwrap();
                    }
                }
            }
        }
    }
}
