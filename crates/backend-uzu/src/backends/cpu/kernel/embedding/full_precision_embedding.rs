use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(FullPrecisionEmbeddingLookup)]
#[variants(T, f32, f16, bf16)]
pub fn full_precision_embedding_lookup<T: ArrayElement + Float>(
    token_ids: *const u64,
    weights: *const T,
    output: *mut T,
    batch_size: u32,
    vocab_size: u32,
    model_dim: u32,
    input_scale: f32,
) {
    for batch_idx in 0..batch_size {
        let token_id = unsafe { *token_ids.add(batch_idx as usize) };
        for dim_idx in 0..model_dim {
            let output_idx = (batch_idx * model_dim + dim_idx) as usize;
            if token_id >= vocab_size as u64 {
                unsafe { *output.add(output_idx) = T::zero() };
            } else {
                unsafe {
                    let weights_idx = (token_id * (model_dim as u64) + dim_idx as u64) as usize;
                    *output.add(output_idx) = *weights.add(weights_idx) * T::from(input_scale).unwrap();
                };
            }
        }
    }
}
