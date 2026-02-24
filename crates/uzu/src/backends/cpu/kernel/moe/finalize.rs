use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoeFinalize)]
#[variants(T, f32, f16, bf16)]
pub fn moe_finalize<T: ArrayElement + Float>(
    #[allow(unused)] tok2row: *const i32,
    #[allow(unused)] probs: *const T,
    #[allow(unused)] y_partial: *const T,
    #[allow(unused)] y: *mut T,
    #[allow(unused)] t_count: u32,
    #[allow(unused)] d_model: u32,
    #[allow(unused)] k_input: u32,
) {
    todo!()
}
