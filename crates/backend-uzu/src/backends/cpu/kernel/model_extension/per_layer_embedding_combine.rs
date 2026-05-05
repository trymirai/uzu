use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(PerLayerEmbeddingCombine)]
#[variants(T, f32, f16, bf16)]
#[variants(ScaleT, f32, f16, bf16)]
pub fn per_layer_embedding_combine<T: ArrayElement + Float, ScaleT: ArrayElement + Float>(
    token_ple: *const T,
    model_ple: *const T,
    scales: *const ScaleT,
    combined: *mut T,
    suffix_length: u32,
    num_layers: u32,
    ple_dim: u32,
    model_projection_scale: f32,
    input_scale: f32,
    epsilon: f32,
    scale_offset: f32,
) {
    let suffix_length = suffix_length as usize;
    let num_layers = num_layers as usize;
    let ple_dim = ple_dim as usize;
    let total_ple_dim = num_layers * ple_dim;

    for token in 0..suffix_length {
        for layer in 0..num_layers {
            let offset = token * total_ple_dim + layer * ple_dim;
            let mut sum_sq = 0.0f32;
            for dim in 0..ple_dim {
                let value = unsafe { (*model_ple.add(offset + dim)).to_f32().unwrap() } * model_projection_scale;
                sum_sq += value * value;
            }
            let rms_inv = (sum_sq / ple_dim as f32 + epsilon).sqrt().recip();
            for dim in 0..ple_dim {
                let model_value = unsafe { (*model_ple.add(offset + dim)).to_f32().unwrap() } * model_projection_scale;
                let scale = unsafe { (*scales.add(dim)).to_f32().unwrap() } + scale_offset;
                let token_value = unsafe { (*token_ple.add(offset + dim)).to_f32().unwrap() };
                let value = (token_value + model_value * rms_inv * scale) * input_scale;
                unsafe { *combined.add(offset + dim) = T::from(value).unwrap() };
            }
        }
    }
}
