use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AudioAdd)]
#[variants(T, f32, f16, bf16)]
pub fn audio_add<T: ArrayElement + Float>(
    a: *const T,
    b: *const T,
    out: *mut T,
    n: i32,
) {
    let _ = (a, b, out, n);
    todo!()
}
