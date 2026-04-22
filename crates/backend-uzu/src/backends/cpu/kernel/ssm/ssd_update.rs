use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{ArrayElement, backends::common::gpu_types::ActivationType};

#[kernel(SSDUpdate)]
#[variants(T, f32, f16, bf16)]
pub fn ssd_update<T: ArrayElement + Float>(
    x: *const T,
    dt_raw: *const T,
    b: *const T,
    c: *const T,
    d: *const T,
    z: *const T,
    #[optional(!state_in_place)] state: Option<*const T>,
    y: *mut T,
    next_state: *mut T,
    group_size: u32,
    state_size: u32,
    x_strides: &[u32; 3],
    dt_strides: &[u32; 2],
    cb_strides: &[u32; 3],
    state_strides: &[u32; 4],
    b_size: u32,
    h_size: u32,
    dh_size: u32,
    #[specialize] state_in_place: bool,
) {
    let group_size = group_size as usize;
    let state_size = state_size as usize;
    let b_size = b_size as usize;
    let h_size = h_size as usize;
    let dh_size = dh_size as usize;

    let state = match state_in_place {
        true => next_state,
        false => state.unwrap(),
    };

    unsafe {
        for b_idx in 0..b_size {
            for h_idx in 0..h_size {
                for dh_idx in 0..dh_size {
                    let cb_start_idx = b_idx * cb_strides[0] as usize + (h_idx / group_size) * cb_strides[1] as usize;
                    let x_idx =
                        b_idx * x_strides[0] as usize + h_idx * x_strides[1] as usize + dh_idx * x_strides[2] as usize;
                    let dt_idx = b_idx * dt_strides[0] as usize + h_idx * dt_strides[1] as usize;
                    let state_start_idx = b_idx * state_strides[0] as usize
                        + h_idx * state_strides[1] as usize
                        + dh_idx * state_strides[2] as usize;

                    let this_x = *x.add(x_idx);
                    let dt_raw_val = *dt_raw.add(dt_idx);
                    let this_dt = ActivationType::SOFTPLUS.activate(dt_raw_val);
                    let this_decay = T::from((-this_dt.to_f32().unwrap()).exp()).unwrap();
                    let this_d = *d.add(h_idx);
                    let this_z = ActivationType::SILU.activate(*z.add(x_idx));

                    let mut temp = T::zero();
                    for i in 0..state_size {
                        let cb_idx = cb_start_idx + i;
                        let state_idx = state_start_idx + i;
                        let this_new_state = *state.add(state_idx) * this_decay + *b.add(cb_idx) * this_x;
                        *next_state.add(state_idx) = this_new_state;
                        temp = temp + this_new_state * *c.add(cb_idx);
                    }
                    temp = temp + this_d * this_x;
                    temp = temp * this_z;
                    *y.add(x_idx) = temp;
                }
            }
        }
    }
}
