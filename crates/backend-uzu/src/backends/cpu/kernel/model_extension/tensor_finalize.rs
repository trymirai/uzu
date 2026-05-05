use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(TensorFinalize)]
#[variants(T, f32, f16, bf16)]
pub fn tensor_finalize<T: ArrayElement + Float>(
    shortcut: *mut T,
    main: *mut T,
    #[optional(has_scalar)] scalar: Option<*const T>,
    length: u32,
    #[specialize] has_scalar: bool,
) {
    let scalar = match (has_scalar, scalar) {
        (true, Some(scalar)) => unsafe { (*scalar).to_f32().unwrap() },
        (false, _) => 1.0,
        (true, None) => return,
    };

    for i in 0..length as usize {
        unsafe {
            let value = ((*shortcut.add(i)).to_f32().unwrap() + (*main.add(i)).to_f32().unwrap()) * scalar;
            *shortcut.add(i) = T::from(value).unwrap();
            *main.add(i) = T::zero();
        }
    }
}
