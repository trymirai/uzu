use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{ArrayElement, backends::common::gpu_types::ActivationType};

#[kernel(SSDPrefill64)]
#[variants(T, f32, f16, bf16)]
pub fn ssd_prefill64<T: ArrayElement + Float>(
    #[allow(unused)] x: *const T,
    #[allow(unused)] dt_raw: *const T,
    #[allow(unused)] b: *const T,
    #[allow(unused)] c: *const T,
    #[allow(unused)] d: *const T,
    #[allow(unused)] z: *const T,
    #[allow(unused)] state: *mut T,
    #[allow(unused)] y: *mut T,
    #[allow(unused)] suffix_len: u32,
    #[allow(unused)] group_size: u32,
    #[allow(unused)] state_size: u32,
    #[allow(unused)] x_strides: &[u32],
    #[allow(unused)] dt_strides: &[u32],
    #[allow(unused)] cb_strides: &[u32],
    #[allow(unused)] state_strides: &[u32],
    #[allow(unused)] num_heads: u32,
    #[allow(unused)] head_dim: u32,
) {
    todo!()
}

#[kernel(SSDPrefill)]
#[variants(T, f32, f16, bf16)]
pub fn ssd_prefill<T: ArrayElement + Float>(
    #[allow(unused)] x: *const T,
    #[allow(unused)] dt_raw: *const T,
    #[allow(unused)] b: *const T,
    #[allow(unused)] c: *const T,
    #[allow(unused)] d: *const T,
    #[allow(unused)] z: *const T,
    #[allow(unused)] state: *mut T,
    #[allow(unused)] y: *mut T,
    #[allow(unused)] suffix_len: u32,
    #[allow(unused)] group_size: u32,
    #[allow(unused)] state_size: u32,
    #[allow(unused)] x_strides: &[u32],
    #[allow(unused)] dt_strides: &[u32],
    #[allow(unused)] cb_strides: &[u32],
    #[allow(unused)] state_strides: &[u32],
    #[allow(unused)] num_heads: u32,
    #[allow(unused)] head_dim: u32,
) {
    todo!()
}

#[kernel(SSDPrefillSequential)]
#[variants(T, f32, f16, bf16)]
pub fn ssd_prefill_sequential<T: ArrayElement + Float>(
    x: *const T,
    dt_raw: *const T,
    b: *const T,
    c: *const T,
    d: *const T,
    z: *const T,
    state: *mut T,
    y: *mut T,
    suffix_len: u32,
    group_size: u32,
    state_size: u32,
    x_strides: &[u32],
    dt_strides: &[u32],
    cb_strides: &[u32],
    state_strides: &[u32],
    channels: u32,
    head_dim: u32,
) {
    let suffix_len = suffix_len as usize;
    let group_size = group_size as usize;
    let state_size = state_size as usize;
    let channels = channels as usize;
    let head_dim = head_dim as usize;

    let total_pairs = suffix_len * channels * head_dim;
    let safe_group = group_size.max(1) as usize;

    unsafe {
        for h in 0..channels {
            let group_idx = h / safe_group;
            for dh in 0..head_dim {
                let state_base = h * state_strides[0] as usize + dh * state_strides[1] as usize;
                for token in 0..suffix_len {
                    let x_idx = token * x_strides[0] as usize + h * x_strides[1] as usize + dh * x_strides[2] as usize;
                    let dt_idx = token * dt_strides[0] as usize + h * dt_strides[1] as usize;
                    let cb_base = token * cb_strides[0] as usize + group_idx * cb_strides[1] as usize;

                    let x_val = *x.add(x_idx);
                    let dt_raw = *dt_raw.add(dt_idx);
                    let dt_val = ActivationType::SOFTPLUS.activate(dt_raw);
                    let decay_val = (-dt_val).exp();
                    let dt_scaled_input = x_val;

                    let mut acc = *d.add(h) * x_val;
                    for s in 0..state_size {
                        let state_idx = state_base + s * state_strides[2] as usize;
                        let cb_idx = cb_base + s * cb_strides[2] as usize;
                        let b_coeff = *b.add(cb_idx);
                        let c_coeff = *c.add(cb_idx);
                        let new_state = decay_val * *state.add(state_idx) + dt_scaled_input * b_coeff;
                        *state.add(state_idx) = new_state;
                        acc = acc + new_state * c_coeff;
                    }

                    let gate = ActivationType::SILU.activate(*z.add(x_idx));
                    *y.add(x_idx) = acc * gate;
                }
            }
        }
    }
}
