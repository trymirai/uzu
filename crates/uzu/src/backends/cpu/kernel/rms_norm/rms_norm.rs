use dsl::kernel;
use half::{bf16, f16};
use num_traits::{Float, ToPrimitive};

use crate::{ArrayElement, DataType};

#[kernel(RMSNorm)]
#[variants(InputT, f32, f16, bf16)]
#[variants(ScaleT, f32, f16, bf16)]
#[variants(OutputT, f32, f16, bf16)]
#[variants(AccumT, f32, f16)]
pub fn rms_norm<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float,
    AccumT: ArrayElement + Float,
>(
    #[optional(!in_place)] input: Option<*const InputT>,
    scales: *const ScaleT,
    output: *mut OutputT,
    batch_size: u32,
    element_count: u32,
    epsilon: f32,
    scale_offset: f32,
    full_layer: bool,
    #[specialize] in_place: bool,
) {
    let input = match in_place {
        true => output as *const InputT,
        false => input.unwrap(),
    };

    let element_count = element_count as usize;
    let input_size = (batch_size as usize) * element_count;
    let scales_size = input_size;
    let epsilon = AccumT::from(epsilon).unwrap();
    let scale_offset = AccumT::from(scale_offset).unwrap();
    let element_count_accum = AccumT::from(element_count).unwrap();

    for batch in 0..(batch_size as usize) {
        let batch_offset = batch * element_count;

        // Compute rms_inv
        let mut sum_sq = AccumT::zero();
        for i in 0..element_count {
            let input_val = unsafe { AccumT::from(*input.add(batch_offset + i)).unwrap() };
            sum_sq = sum_sq + input_val * input_val;
        }
        let mean_sq: AccumT = sum_sq / element_count_accum;
        let rms_inv = AccumT::from((mean_sq + epsilon).to_f32().unwrap().sqrt().recip()).unwrap();

        // Normalization and scaling
        for i in 0..element_count {
            let input_val = unsafe { AccumT::from(*input.add(batch_offset + i)).unwrap() };
            let scale_val = unsafe { AccumT::from(*scales.add(i)).unwrap() };
            let result: OutputT = if full_layer {
                // Full-layer: keep everything in accumulation precision
                let normalized: AccumT = input_val * rms_inv;
                let scale_with_offset: AccumT = scale_val + scale_offset;
                OutputT::from(normalized * scale_with_offset).unwrap()
            } else {
                // Only-normalization: cast down to output precision for the scale multiply
                let normalized: AccumT = input_val * rms_inv;
                let normalized_out = OutputT::from(normalized).unwrap();
                let scale_with_offset_out = OutputT::from(scale_val + scale_offset).unwrap();
                normalized_out * scale_with_offset_out
            };
            unsafe { *output.add(batch_offset + i) = result };
        }
    }
}
