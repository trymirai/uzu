use dsl::kernel;
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoeCountsOffsetsFused)]
pub fn moe_counts_offsets_fused(
    #[allow(unused)] topk_ids: *const i32,
    #[allow(unused)] offsets: *mut u32,
    #[allow(unused)] sum_k_out: *mut u32,
    #[allow(unused)] partials: *mut u32,
    #[allow(unused)] t_input: u32,
    #[allow(unused)] e_input: u32,
    #[allow(unused)] k_input: u32,
) {
    todo!()
}
