use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(SoftCap)]
#[variants(T, f32, f16, bf16)]
pub fn soft_cap<T: ArrayElement + Float>(
    #[optional(!in_place)] input: Option<*const T>,
    output: *mut T,
    length: u32,
    cap: f32,
    #[specialize] in_place: bool,
) {
    let input = match (in_place, input) {
        (true, _) => output as *const T,
        (false, Some(input)) => input,
        (false, None) => return,
    };

    for i in 0..length as usize {
        unsafe {
            let value = (*input.add(i)).to_f32().unwrap();
            *output.add(i) = T::from(cap * (value / cap).tanh()).unwrap();
        }
    }
}
