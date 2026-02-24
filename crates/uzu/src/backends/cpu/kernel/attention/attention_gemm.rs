use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AttentionGemm)]
#[variants(T, f32, f16, bf16)]
#[variants(BK, 16, 32)]
#[variants(BD, 64, 128, 256)]
pub fn attention_gemm<T: ArrayElement + Float, const BK: u32, const BD: u32>(
    #[allow(unused)] q: *const T,
    #[allow(unused)] k: *const T,
    #[allow(unused)] v: *const T,
    #[allow(unused)] o: *mut T,
    #[allow(unused)] params: crate::backends::common::gpu_types::attention::AttnParams,
    #[allow(unused)]
    #[optional(has_mask)]
    mask_params: Option<crate::backends::common::gpu_types::attention::AttnMaskParams>,
    #[allow(unused)]
    #[optional(has_mask)]
    mask: Option<*const T>,
    #[allow(unused)]
    #[optional(has_sinks)]
    sinks: Option<*const f32>,
    #[allow(unused)] num_heads: u32,
    #[allow(unused)] suffix_length: u32,
    #[allow(unused)]
    #[specialize]
    align_q: bool,
    #[allow(unused)]
    #[specialize]
    align_k: bool,
    #[allow(unused)]
    #[specialize]
    do_causal: bool,
    #[allow(unused)]
    #[specialize]
    has_mask: bool,
    #[allow(unused)]
    #[specialize]
    has_sinks: bool,
) {
    todo!()
}
