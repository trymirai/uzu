use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(TopK)]
#[variants(T, f32, f16, bf16)]
pub fn top_k<T: ArrayElement + Float>(
    #[allow(unused)]
    #[optional(!in_place)]
    logits: Option<*const T>,
    #[allow(unused)] processed_logits: *mut T,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] vocab_size: u32,
    #[allow(unused)] top_k: u32,
    #[allow(unused)]
    #[specialize]
    in_place: bool,
) {
    todo!()
}
