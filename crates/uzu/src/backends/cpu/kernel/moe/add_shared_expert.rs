use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoeAddSharedExpert)]
#[variants(T, f32, f16, bf16)]
pub fn moe_add_shared_expert<T: ArrayElement + Float>(
    shared_out: *const T,
    gate_logits: *const T,
    y: *mut T,
    t_count: u32,
    d_model: u32,
    num_shared: u32,
) {
    let t_count = t_count as usize;
    let d_model = d_model as usize;
    let num_shared = num_shared as usize;

    unsafe {
        for t in 0..t_count {
            let mut gw = 0f32;
            for i in 0..num_shared {
                let lg = (*gate_logits.add(t * num_shared + i)).to_f32().unwrap();
                gw += 1.0 / (1.0 + (-lg).exp());
            }
            for f in 0..d_model {
                let idx = t * d_model + f;
                let yv = (*y.add(idx)).to_f32().unwrap();
                let sv = (*shared_out.add(idx)).to_f32().unwrap();
                *y.add(idx) = T::from(yv + gw * sv).unwrap();
            }
        }
    }
}
