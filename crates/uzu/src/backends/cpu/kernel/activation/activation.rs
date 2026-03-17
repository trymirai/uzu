use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(Activation)]
#[variants(T, f16, f32, bf16)]
pub fn activation<T: ArrayElement + Float>(
    #[allow(unused)]
    #[optional(!in_place)]
    input: Option<*const T>,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] n: u32,
    #[allow(unused)] act_type: u32,
    #[allow(unused)]
    #[specialize]
    in_place: bool,
) {
    let src: *const T = if in_place {
        output as *const T
    } else {
        input.unwrap()
    };

    let act = super::ActivationType::try_from(act_type).unwrap();

    for i in 0..n as usize {
        let val = unsafe { *src.add(i) };
        unsafe { *output.add(i) = super::activate(val, &act) };
    }
}
