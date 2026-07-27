use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

// Pre-compute L2-normalized q/k, beta, and decay for all tokens.
#[kernel(DeltaNetPrefillPrep)]
#[variants(T, f32, bf16)]
#[variants(QKT, f32, bf16)]
#[variants(HEAD_K_DIM, 128)]
pub fn delta_net_prefill_prep<T: ArrayElement + Float, QKT: ArrayElement + Float, const HEAD_K_DIM: u32>(
    in_proj: *const T,
    a_log: *const f32,
    dt_bias: *const f32,
    q_norm_out: *mut QKT,
    k_norm_out: *mut QKT,
    #[optional(write_compact_v)] compact_v_out: Option<*mut T>,
    beta_out: *mut f32,
    decay_out: *mut f32,
    num_v_heads: u32,
    num_k_heads: u32,
    key_dim: u32,
    value_dim: u32,
    suffix_len: u32,
    #[specialize] write_log_decay: bool,
    #[specialize] write_compact_v: bool,
) {
    assert_eq!(compact_v_out.is_some(), write_compact_v, "compact V output presence mismatch");

    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let head_k_dim = HEAD_K_DIM as usize;
    let key_dim = key_dim as usize;
    let value_dim = value_dim as usize;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let suffix_len = suffix_len as usize;
    let groups_per_head = num_v_heads / num_k_heads;

    for token in 0..suffix_len {
        let tok_offset = token * total_proj_dim;

        if let Some(compact_v_out) = compact_v_out {
            unsafe {
                in_proj
                    .add(tok_offset + 2 * key_dim)
                    .copy_to_nonoverlapping(compact_v_out.add(token * value_dim), value_dim)
            };
        }

        for hk in 0..num_k_heads {
            // Load and L2-normalize q
            let q_off = tok_offset + hk * head_k_dim;
            let mut q_sq = 0.0f32;
            for j in 0..head_k_dim {
                let v = unsafe { (*in_proj.add(q_off + j)).to_f32().unwrap() };
                q_sq += v * v;
            }
            let q_inv = 1.0 / (q_sq + 1e-6).sqrt();
            let q_scale = 1.0 / (head_k_dim as f32).sqrt();
            for j in 0..head_k_dim {
                let v = unsafe { (*in_proj.add(q_off + j)).to_f32().unwrap() };
                unsafe {
                    *q_norm_out.add(token * key_dim + hk * head_k_dim + j) = QKT::from(v * q_inv * q_scale).unwrap()
                };
            }

            // Load and L2-normalize k
            let k_off = tok_offset + key_dim + hk * head_k_dim;
            let mut k_sq = 0.0f32;
            for j in 0..head_k_dim {
                let v = unsafe { (*in_proj.add(k_off + j)).to_f32().unwrap() };
                k_sq += v * v;
            }
            let k_inv = 1.0 / (k_sq + 1e-6).sqrt();
            for j in 0..head_k_dim {
                let v = unsafe { (*in_proj.add(k_off + j)).to_f32().unwrap() };
                unsafe { *k_norm_out.add(token * key_dim + hk * head_k_dim + j) = QKT::from(v * k_inv).unwrap() };
            }

            // Beta and decay for each v-head of this k-head
            for group in 0..groups_per_head {
                let hv = hk * groups_per_head + group;

                let beta_raw = unsafe { (*in_proj.add(tok_offset + conv_dim + value_dim + hv)).to_f32().unwrap() };
                let beta = 1.0 / (1.0 + (-beta_raw).exp());

                let a_log_val = unsafe { *a_log.add(hv) };
                let dt_bias_val = unsafe { *dt_bias.add(hv) };
                let a_raw =
                    unsafe { (*in_proj.add(tok_offset + conv_dim + value_dim + num_v_heads + hv)).to_f32().unwrap() };
                let sp_in = a_raw + dt_bias_val;
                let sp = if sp_in > 20.0 {
                    sp_in
                } else {
                    (1.0 + sp_in.exp()).ln()
                };
                let log_decay = -a_log_val.exp() * sp;

                unsafe {
                    *beta_out.add(token * num_v_heads + hv) = beta;
                    *decay_out.add(token * num_v_heads + hv) = if write_log_decay {
                        log_decay
                    } else {
                        log_decay.exp()
                    };
                };
            }
        }
    }
}
