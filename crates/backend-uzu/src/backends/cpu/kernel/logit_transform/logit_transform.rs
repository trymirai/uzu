use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

#[kernel(LogitTransform)]
#[variants(T, f32, bf16)]
pub fn logit_transform<T: ArrayElement + Float>(
    logits: *mut T,
    length: u32,
    scale: f32,
    soft_cap: f32,
    #[specialize] has_soft_cap: bool,
) {
    let length = length as usize;
    unsafe {
        for position in 0..length {
            let mut value = (*logits.add(position)).to_f32().unwrap() * scale;
            if has_soft_cap {
                value = (value / soft_cap).tanh() * soft_cap;
            }
            *logits.add(position) = T::from(value).unwrap();
        }
    }
}
