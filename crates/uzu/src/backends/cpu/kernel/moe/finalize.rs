use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoeFinalize)]
#[variants(T, f32, f16, bf16)]
pub fn moe_finalize<T: ArrayElement + Float>(
    tok2row: *const i32,
    probs: *const T,
    y_partial: *const T,
    y: *mut T,
    t_count: u32,
    d_model: u32,
    k_input: u32,
) {
    let t_count = t_count as usize;
    let d_model = d_model as usize;
    let k_input = k_input as usize;

    unsafe {
        for ti in 0..t_count {
            for f in 0..d_model {
                let mut acc = 0f32;
                for kk in 0..k_input {
                    let idx = ti * k_input + kk;
                    let row = *tok2row.add(idx);
                    if row >= 0 {
                        let rowu = row as usize;
                        acc += (*probs.add(idx)).to_f32().unwrap()
                            * (*y_partial.add(rowu * d_model + f)).to_f32().unwrap();
                    }
                }
                *y.add(ti * d_model + f) = T::from(acc).unwrap();
            }
        }
    }
}
