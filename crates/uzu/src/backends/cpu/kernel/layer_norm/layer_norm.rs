use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(LayerNorm)]
#[variants(IN, f32, f16, bf16)]
#[variants(SC, f32, f16, bf16)]
#[variants(OUT, f32, f16, bf16)]
#[variants(ACC, f32)]
pub fn layer_norm<
    IN: ArrayElement + Float,
    SC: ArrayElement + Float,
    OUT: ArrayElement + Float,
    ACC: ArrayElement + Float,
>(
    #[allow(unused)]
    #[optional(!in_place)]
    input: Option<*const IN>,
    #[allow(unused)] scales: *const SC,
    #[allow(unused)] output: *mut OUT,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] model_dim: u32,
    #[allow(unused)] epsilon: f32,
    #[allow(unused)] scale_offset: f32,
    #[allow(unused)] full_layer: u32,
    #[allow(unused)]
    #[specialize]
    in_place: bool,
) {
    let src: *const IN = if in_place {
        output as *const OUT as *const IN
    } else {
        input.unwrap()
    };

    for batch in 0..batch_size {
        let offset = (batch * model_dim) as usize;

        // Compute mean
        let mut sum = 0.0f32;
        for i in 0..model_dim as usize {
            sum += unsafe { (*src.add(offset + i)).to_f32().unwrap() };
        }
        let mean = sum / model_dim as f32;

        // Compute variance
        let mut var_sum = 0.0f32;
        for i in 0..model_dim as usize {
            let diff = unsafe { (*src.add(offset + i)).to_f32().unwrap() } - mean;
            var_sum += diff * diff;
        }
        let variance = var_sum / model_dim as f32;
        let inv_std = 1.0f32 / (variance + epsilon).sqrt();

        // Normalize and scale
        for i in 0..model_dim as usize {
            let x = unsafe { (*src.add(offset + i)).to_f32().unwrap() };
            let normalized = (x - mean) * inv_std;

            let result = if full_layer != 0 {
                let scale = unsafe { (*scales.add(i)).to_f32().unwrap() } + scale_offset;
                normalized * scale
            } else {
                normalized
            };

            unsafe { *output.add(offset + i) = OUT::from(result).unwrap() };
        }
    }
}
