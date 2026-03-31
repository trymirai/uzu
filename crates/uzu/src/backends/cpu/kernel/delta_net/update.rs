use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{ArrayElement, backends::common::gpu_types::ActivationType};

// Single-token delta net update: decay state, apply delta rule, RMSNorm + SiLU gate.
// Steps per v-head:
//   1. L2-normalize and scale q, k
//   2. Compute decay from a_log and softplus(a_raw + dt_bias)
//   3. For each v-dim: read state, compute output, update state with delta rule
//   4. RMSNorm output, gate with SiLU(z), write result
#[kernel(DeltaNetUpdate)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_K_DIM, 128)]
pub fn delta_net_update<T: ArrayElement + Float, const HEAD_K_DIM: u32>(
    in_proj: *const T,
    a_log: *const T,
    dt_bias: *const T,
    norm_weight: *const T,
    state: *mut T,
    out: *mut T,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    key_dim: u32,
    value_dim: u32,
    norm_epsilon: f32,
) {
    let state_ptr = state as *const T;

    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    debug_assert!(num_v_heads % num_k_heads == 0, "num_v_heads must be a multiple of num_k_heads");
    let head_k_dim = HEAD_K_DIM as usize;
    let head_v_dim = head_v_dim as usize;
    let key_dim = key_dim as usize;
    let value_dim = value_dim as usize;
    let conv_dim = 2 * key_dim + value_dim;

    // in_proj layout (after conv):
    // [0..key_dim): q (conv'd + SiLU)
    // [key_dim..2*key_dim): k (conv'd + SiLU)
    // [2*key_dim..2*key_dim+value_dim): v (conv'd + SiLU)
    // [conv_dim..conv_dim+value_dim): z (raw)
    // [conv_dim+value_dim..conv_dim+value_dim+num_v_heads): beta_raw (raw)
    // [conv_dim+value_dim+num_v_heads..conv_dim+value_dim+2*num_v_heads): a_raw (raw)

    for hv in 0..num_v_heads {
        let hk = hv / (num_v_heads / num_k_heads);

        // Load q and k for this head
        let q_offset = hk * head_k_dim;
        let k_offset = key_dim + hk * head_k_dim;

        let mut q = vec![0.0f32; head_k_dim];
        let mut k = vec![0.0f32; head_k_dim];
        for j in 0..head_k_dim {
            q[j] = unsafe { (*in_proj.add(q_offset + j)).to_f32().unwrap() };
            k[j] = unsafe { (*in_proj.add(k_offset + j)).to_f32().unwrap() };
        }

        // L2 normalize q and k
        let q_norm_sq: f32 = q.iter().map(|x| x * x).sum();
        let k_norm_sq: f32 = k.iter().map(|x| x * x).sum();
        let q_inv_norm = 1.0 / (q_norm_sq + 1e-6).sqrt();
        let k_inv_norm = 1.0 / (k_norm_sq + 1e-6).sqrt();
        for j in 0..head_k_dim {
            q[j] *= q_inv_norm;
            k[j] *= k_inv_norm;
        }

        // Scale q by head_k_dim^-0.5
        let q_scale = 1.0 / (head_k_dim as f32).sqrt();
        for j in 0..head_k_dim {
            q[j] *= q_scale;
        }

        // Compute gating
        let beta_raw = unsafe { (*in_proj.add(conv_dim + value_dim + hv)).to_f32().unwrap() };
        let beta = 1.0 / (1.0 + (-beta_raw).exp()); // sigmoid

        let a_raw = unsafe { (*in_proj.add(conv_dim + value_dim + num_v_heads + hv)).to_f32().unwrap() };
        let a_log_val = unsafe { (*a_log.add(hv)).to_f32().unwrap() };
        let dt_bias_val = unsafe { (*dt_bias.add(hv)).to_f32().unwrap() };

        // softplus(a_raw + dt_bias)
        let sp_input = a_raw + dt_bias_val;
        let sp = if sp_input > 20.0 {
            sp_input
        } else {
            (1.0 + sp_input.exp()).ln()
        };
        let g = -a_log_val.exp() * sp;
        let decay = g.exp();

        // dot(k, q)
        let kq_dot: f32 = k.iter().zip(q.iter()).map(|(ki, qi)| ki * qi).sum();

        // State layout: [num_v_heads, head_v_dim, head_k_dim]
        let mut o = vec![0.0f32; head_v_dim];

        for i in 0..head_v_dim {
            let v_i = unsafe { (*in_proj.add(2 * key_dim + hv * head_v_dim + i)).to_f32().unwrap() };
            let state_row = hv * head_v_dim * head_k_dim + i * head_k_dim;

            // Pass 1: read state row S[i,:], compute dot(S[i,:], q) and dot(S[i,:], k)
            let mut sq_acc = 0.0f32;
            let mut sk_acc = 0.0f32;
            for j in 0..head_k_dim {
                let s = unsafe { (*state_ptr.add(state_row + j)).to_f32().unwrap() };
                sq_acc += s * q[j];
                sk_acc += s * k[j];
            }

            let retrieved_i = decay * sk_acc;
            let delta_i = beta * (v_i - retrieved_i);
            o[i] = decay * sq_acc + delta_i * kq_dot;

            // Pass 2: update state row S[i,:]
            for j in 0..head_k_dim {
                let s = unsafe { (*state_ptr.add(state_row + j)).to_f32().unwrap() };
                unsafe {
                    *state.add(state_row + j) = T::from(decay * s + k[j] * delta_i).unwrap();
                }
            }
        }

        // RMSNorm over o
        let sumsq: f32 = o.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / (sumsq / head_v_dim as f32 + norm_epsilon).sqrt();

        // Apply RMSNorm + SiLU gate and write output
        for i in 0..head_v_dim {
            let norm_w = unsafe { (*norm_weight.add(i)).to_f32().unwrap() };
            let z_i = unsafe { (*in_proj.add(conv_dim + hv * head_v_dim + i)).to_f32().unwrap() };
            let z_silu = ActivationType::SILU.activate(z_i);
            let final_val = o[i] * inv_rms * norm_w * z_silu;
            unsafe {
                *out.add(hv * head_v_dim + i) = T::from(final_val).unwrap();
            }
        }
    }
}
