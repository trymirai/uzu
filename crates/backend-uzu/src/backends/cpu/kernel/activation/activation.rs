use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(Activation)]
#[variants(T, f16, f32, bf16)]
pub fn activation<T: ArrayElement + Float>(
    #[optional(!in_place)] input: Option<*const T>,
    output: *mut T,
    n: u32,
    act_type: crate::backends::common::gpu_types::activation_type::ActivationType,
    #[specialize] in_place: bool,
) {
    let input = match in_place {
        true => output,
        false => input.unwrap(),
    };
    unsafe {
        for i in 0..n as usize {
            *output.add(i) = act_type.activate(*input.add(i));
        }
    }
}
