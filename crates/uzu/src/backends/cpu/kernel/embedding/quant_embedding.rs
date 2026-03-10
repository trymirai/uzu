use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(QuantizedEmbeddingLookup)]
#[variants(T, f32, f16, bf16)]
pub fn quantized_embedding_lookup<T: ArrayElement + Float>(
    #[allow(unused)] token_ids: *const u64,
    #[allow(unused)] weights: *const u8,
    #[allow(unused)] scales: *const T,
    #[allow(unused)] biases: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] vocab_size: u32,
    #[allow(unused)] model_dim: u32,
    #[allow(unused)] group_size: u32,
    #[allow(unused)] input_scale: f32,
    #[allow(unused)] quant_mode: u32,
) {
    todo!()
}
