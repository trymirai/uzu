use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AudioAdd)]
#[variants(T, f32, f16, bf16)]
pub fn audio_add<T: ArrayElement + Float>(
    #[allow(unused)] a: *const T,
    #[allow(unused)] b: *const T,
    #[allow(unused)] out: *mut T,
    #[allow(unused)] n: i32,
) {
    todo!()
}
