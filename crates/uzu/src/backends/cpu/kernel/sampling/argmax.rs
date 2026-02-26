use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(ArgmaxSingle)]
#[variants(T, f32, f16, bf16)]
pub fn argmax_single<T: ArrayElement + Float>(
    #[allow(unused)] logits_data: *const T,
    #[allow(unused)] final_tokens: *mut u32,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] vocab_size: u32,
) {
    todo!()
}

#[kernel(ArgmaxMain)]
#[variants(T, f32, f16, bf16)]
pub fn argmax_main<T: ArrayElement + Float>(
    #[allow(unused)] logits_data: *const T,
    #[allow(unused)] partial_results: *mut crate::backends::common::gpu_types::argmax::ArgmaxPair,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] vocab_size: u32,
) {
    todo!()
}

#[kernel(ArgmaxFinal)]
pub fn argmax_final(
    #[allow(unused)] partial_results: *const crate::backends::common::gpu_types::argmax::ArgmaxPair,
    #[allow(unused)] final_tokens: *mut u32,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] vocab_size: u32,
) {
    todo!()
}
