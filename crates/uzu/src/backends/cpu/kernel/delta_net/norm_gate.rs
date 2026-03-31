use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{ArrayElement, backends::common::gpu_types::ActivationType};

// In-place RMSNorm + SiLU gate for DeltaNet prefill output.
//
// in_out:      [suffix_len, value_dim] — raw input, overwritten with final output
// in_proj:     [suffix_len, total_proj_dim] — for reading z values
// norm_weight: [head_v_dim]
//
// Per (token, head): in_out[i] = in_out[i] * inv_rms * norm_weight[i] * silu(z[i])
// where inv_rms = rsqrt(mean(in_out^2) + epsilon)
// and z is at in_proj[token * total_proj_dim + conv_dim + hv * head_v_dim + i]
#[kernel(DeltaNetNormGate)]
#[variants(T, f32, f16, bf16)]
pub fn delta_net_norm_gate<T: ArrayElement + Float>(
    in_out: *mut T,
    in_proj: *const T,
    norm_weight: *const T,
    num_v_heads: u32,
    head_v_dim: u32,
    value_dim: u32,
    conv_dim: u32,
    total_proj_dim: u32,
    norm_epsilon: f32,
    suffix_len: u32,
) {
    let in_out_ptr = in_out as *const T;
    let num_v_heads = num_v_heads as usize;
    let head_v_dim = head_v_dim as usize;
    let value_dim = value_dim as usize;
    let conv_dim = conv_dim as usize;
    let total_proj_dim = total_proj_dim as usize;
    let suffix_len = suffix_len as usize;

    for token in 0..suffix_len {
        for hv in 0..num_v_heads {
            let base = token * value_dim + hv * head_v_dim;

            let mut sumsq = 0.0f32;
            for i in 0..head_v_dim {
                let val = unsafe { (*in_out_ptr.add(base + i)).to_f32().unwrap() };
                sumsq += val * val;
            }
            let inv_rms = 1.0 / (sumsq / head_v_dim as f32 + norm_epsilon).sqrt();

            for i in 0..head_v_dim {
                let o_i = unsafe { (*in_out_ptr.add(base + i)).to_f32().unwrap() };
                let norm_w = unsafe { (*norm_weight.add(i)).to_f32().unwrap() };
                let z_idx = token * total_proj_dim + conv_dim + hv * head_v_dim + i;
                let z_i = unsafe { (*in_proj.add(z_idx)).to_f32().unwrap() };
                let final_val = o_i * inv_rms * norm_w * ActivationType::SILU.activate(z_i);
                unsafe {
                    *in_out.add(base + i) = T::from(final_val).unwrap();
                }
            }
        }
    }
}
