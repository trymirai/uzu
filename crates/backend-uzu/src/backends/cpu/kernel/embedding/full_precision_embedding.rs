use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

#[kernel(FullPrecisionEmbeddingLookup)]
#[variants(T, f32, f16, bf16)]
pub fn full_precision_embedding_lookup<T: ArrayElement + Float>(
    #[allow(unused)] token_ids: *const u32,
    #[allow(unused)] weights: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] vocab_size: u32,
    #[allow(unused)] model_dim: u32,
    #[allow(unused)] input_scale: f32,
) {
    for batch_idx in 0..batch_size {
        let token_id = unsafe { *token_ids.add(batch_idx as usize) };
        for dim_idx in 0..model_dim {
            let output_idx = (batch_idx * model_dim + dim_idx) as usize;
            if token_id >= vocab_size {
                unsafe { *output.add(output_idx) = T::zero() };
            } else {
                unsafe {
                    let weights_idx = token_id as usize * model_dim as usize + dim_idx as usize;
                    *output.add(output_idx) = *weights.add(weights_idx) * T::from(input_scale).unwrap();
                };
            }
        }
    }
}
