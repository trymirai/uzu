use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;
use crate::backends::cpu::kernel::activation::silu_f32;

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
    #[allow(unused)] in_out: *mut T,
    #[allow(unused)] in_proj: *const T,
    #[allow(unused)] norm_weight: *const T,
    #[allow(unused)] num_v_heads: u32,
    #[allow(unused)] head_v_dim: u32,
    #[allow(unused)] value_dim: u32,
    #[allow(unused)] conv_dim: u32,
    #[allow(unused)] total_proj_dim: u32,
    #[allow(unused)] norm_epsilon: f32,
    #[allow(unused)] suffix_len: u32,
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
                let final_val = o_i * inv_rms * nw * silu_f32(z_i);
                unsafe {
                    *in_out.add(base + i) = T::from(final_val).unwrap();
                }
            }
        }
    }
}
