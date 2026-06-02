use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

#[kernel(LogitSoftCap)]
#[variants(T, f32, bf16)]
pub fn logit_soft_cap<T: ArrayElement + Float>(
    logits: *mut T,
    length: u32,
    soft_cap: f32,
) {
    let length = length as usize;
    unsafe {
        for position in 0..length {
            let value = (*logits.add(position)).to_f32().unwrap();
            *logits.add(position) = T::from((value / soft_cap).tanh() * soft_cap).unwrap();
        }
    }
}
