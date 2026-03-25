use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(EmbeddingRowsSum)]
#[variants(T, f32, f16, bf16)]
pub fn embedding_rows_sum<T: ArrayElement + Float>(
    #[allow(unused)] token_indices: *const u32,
    #[allow(unused)] weights: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] num_rows: u32,
    #[allow(unused)] total_rows: u32,
    #[allow(unused)] model_dim: u32,
    #[allow(unused)] codebook_stride: u32,
) {
    todo!()
}
