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
    #[optional(!in_place)] input: Option<*const IN>,
    scales: *const SC,
    output: *mut OUT,
    input_offset_elements: u32,
    batch_size: u32,
    model_dim: u32,
    epsilon: f32,
    scale_offset: f32,
    full_layer: u32,
    #[specialize] in_place: bool,
) {
    let input = match in_place {
        true => output as *const IN,
        false => unsafe { input.unwrap().add(input_offset_elements as usize) },
    };

    let model_dim = model_dim as usize;
    let epsilon = ACC::from(epsilon).unwrap();
    let scale_offset = ACC::from(scale_offset).unwrap();
    let element_count_accum = ACC::from(model_dim).unwrap();
    let full_layer = full_layer != 0;

    for batch in 0..(batch_size as usize) {
        let batch_offset = batch * model_dim;

        // Compute mean
        let mut sum = ACC::zero();
        for i in 0..model_dim {
            let val = unsafe { ACC::from(*input.add(batch_offset + i)).unwrap() };
            sum = sum + val;
        }
        let mean = sum / element_count_accum;

        // Compute variance
        let mut var_sum = ACC::zero();
        for i in 0..model_dim {
            let val = unsafe { ACC::from(*input.add(batch_offset + i)).unwrap() };
            let centered = val - mean;
            var_sum = var_sum + centered * centered;
        }
        let variance = var_sum / element_count_accum;
        let inv_std = ACC::from((variance + epsilon).to_f32().unwrap().sqrt().recip()).unwrap();

        // Normalize and scale
        for i in 0..model_dim {
            let input_val = unsafe { ACC::from(*input.add(batch_offset + i)).unwrap() };
            let scale_val = unsafe { ACC::from(*scales.add(i)).unwrap() };
            let normalized: ACC = (input_val - mean) * inv_std;
            let result: OUT = if full_layer {
                let scale_with_offset: ACC = scale_val + scale_offset;
                OUT::from(normalized * scale_with_offset).unwrap()
            } else {
                let normalized_out = OUT::from(normalized).unwrap();
                let scale_with_offset_out = OUT::from(scale_val + scale_offset).unwrap();
                normalized_out * scale_with_offset_out
            };
            unsafe { *output.add(batch_offset + i) = result };
        }
    }
}
