use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MinP)]
#[variants(T, f32, f16, bf16)]
pub fn min_p<T: ArrayElement + Float>(
    #[allow(unused)] logits: *const T,
    #[allow(unused)] processed_logits: *mut T,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] vocab_size: u32,
    #[allow(unused)] min_p: f32,
) {
    todo!()
}
