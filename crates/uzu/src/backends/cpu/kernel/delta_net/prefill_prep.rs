use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

// Pre-compute L2-normalized q/k, beta, and decay for all tokens.
#[kernel(DeltaNetPrefillPrep)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_K_DIM, 128)]
pub fn delta_net_prefill_prep<T: ArrayElement + Float, const HEAD_K_DIM: u32>(
    in_proj: *const T,
    a_log: *const T,
    dt_bias: *const T,
    q_norm_out: *mut f32,
    k_norm_out: *mut f32,
    beta_out: *mut f32,
    decay_out: *mut f32,
    num_v_heads: u32,
    num_k_heads: u32,
    key_dim: u32,
    value_dim: u32,
    suffix_len: u32,
) {
    let nv = num_v_heads as usize;
    let nk = num_k_heads as usize;
    let dk = HEAD_K_DIM as usize;
    let kd = key_dim as usize;
    let vd = value_dim as usize;
    let conv_dim = 2 * kd + vd;
    let total_proj_dim = conv_dim + vd + nv + nv;
    let sl = suffix_len as usize;
    let groups_per_head = nv / nk;

    for t in 0..sl {
        let tok = t * total_proj_dim;

        for hk in 0..nk {
            // Load and L2-normalize q
            let q_off = tok + hk * dk;
            let mut q_sq = 0.0f32;
            for j in 0..dk {
                let v = unsafe { (*in_proj.add(q_off + j)).to_f32().unwrap() };
                q_sq += v * v;
            }
            let q_inv = 1.0 / (q_sq + 1e-6).sqrt();
            let q_scale = 1.0 / (dk as f32).sqrt();
            for j in 0..dk {
                let v = unsafe { (*in_proj.add(q_off + j)).to_f32().unwrap() };
                unsafe { *q_norm_out.add(t * kd + hk * dk + j) = v * q_inv * q_scale };
            }

            // Load and L2-normalize k
            let k_off = tok + kd + hk * dk;
            let mut k_sq = 0.0f32;
            for j in 0..dk {
                let v = unsafe { (*in_proj.add(k_off + j)).to_f32().unwrap() };
                k_sq += v * v;
            }
            let k_inv = 1.0 / (k_sq + 1e-6).sqrt();
            for j in 0..dk {
                let v = unsafe { (*in_proj.add(k_off + j)).to_f32().unwrap() };
                unsafe { *k_norm_out.add(t * kd + hk * dk + j) = v * k_inv };
            }

            // Beta and decay for each v-head of this k-head
            for g in 0..groups_per_head {
                let hv = hk * groups_per_head + g;

                let beta_raw = unsafe { (*in_proj.add(tok + conv_dim + vd + hv)).to_f32().unwrap() };
                let beta = 1.0 / (1.0 + (-beta_raw).exp());

                let a_log_val = unsafe { (*a_log.add(hv)).to_f32().unwrap() };
                let dt_bias_val = unsafe { (*dt_bias.add(hv)).to_f32().unwrap() };
                let a_raw = unsafe { (*in_proj.add(tok + conv_dim + vd + nv + hv)).to_f32().unwrap() };
                let sp_in = a_raw + dt_bias_val;
                let sp = if sp_in > 20.0 {
                    sp_in
                } else {
                    (1.0 + sp_in.exp()).ln()
                };
                let decay = (-a_log_val.exp() * sp).exp();

                unsafe {
                    *beta_out.add(t * nv + hv) = beta;
                    *decay_out.add(t * nv + hv) = decay;
                };
            }
        }
    }
}
