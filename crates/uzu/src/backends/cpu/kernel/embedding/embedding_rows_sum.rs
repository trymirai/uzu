use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(EmbeddingRowsSum)]
#[variants(T, f32, f16, bf16)]
pub fn embedding_rows_sum<T: ArrayElement + Float>(
    token_indices: *const u32,
    weights: *const T,
    output: *mut T,
    num_rows: u32,
    total_rows: u32,
    model_dim: u32,
    codebook_stride: u32,
) {
    let codebook_stride = codebook_stride as usize;
    let total_rows = total_rows as usize;
    let model_dim = model_dim as usize;

    unsafe {
        for dim_idx in 0..model_dim {
            let mut sum = 0f32;
            for row_idx in 0..num_rows as usize {
                let token = *token_indices.add(row_idx) as usize;
                let row = row_idx * codebook_stride + token;
                if row < total_rows {
                    sum += (*weights.add(row * model_dim + dim_idx)).to_f32().unwrap();
                }
            }
            *output.add(dim_idx) = T::from(sum).unwrap();
        }
    }
}
