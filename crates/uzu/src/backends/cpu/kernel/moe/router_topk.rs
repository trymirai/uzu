use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoeRouterTopK)]
#[variants(ScalarT, f16, bf16, f32)]
pub fn moe_router_top_k<ScalarT: ArrayElement + Float>(
    #[allow(unused)] input: *const ScalarT,
    #[allow(unused)] weight: *const ScalarT,
    #[allow(unused)] bias: *const ScalarT,
    #[allow(unused)] topk_ids: *mut i32,
    #[allow(unused)] topk_probs: *mut ScalarT,
    #[allow(unused)] t: u32,
    #[allow(unused)] d_model: u32,
    #[allow(unused)] e: u32,
    #[allow(unused)] k: u32,
    #[allow(unused)] renorm: bool,
) {
    todo!()
}
