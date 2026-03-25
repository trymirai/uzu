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
// and z is at in_proj[t * total_proj_dim + conv_dim + hv * head_v_dim + i]
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
    let nv = num_v_heads as usize;
    let dv = head_v_dim as usize;
    let vd = value_dim as usize;
    let cd = conv_dim as usize;
    let tpp = total_proj_dim as usize;
    let sl = suffix_len as usize;

    for t in 0..sl {
        for hv in 0..nv {
            let base = t * vd + hv * dv;

            let mut sumsq = 0.0f32;
            for i in 0..dv {
                let val = unsafe { (*in_out_ptr.add(base + i)).to_f32().unwrap() };
                sumsq += val * val;
            }
            let inv_rms = 1.0 / (sumsq / dv as f32 + norm_epsilon).sqrt();

            for i in 0..dv {
                let o_i = unsafe { (*in_out_ptr.add(base + i)).to_f32().unwrap() };
                let nw = unsafe { (*norm_weight.add(i)).to_f32().unwrap() };
                let z_idx = t * tpp + cd + hv * dv + i;
                let z_i = unsafe { (*in_proj.add(z_idx)).to_f32().unwrap() };
                let final_val = o_i * inv_rms * nw * ActivationType::SILU.activate(z_i);
                unsafe {
                    *in_out.add(base + i) = T::from(final_val).unwrap();
                }
            }
        }
    }
}
