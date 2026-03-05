use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(QKNorm)]
#[variants(InputT, f32, f16, bf16)]
#[variants(ScaleT, f32, f16, bf16)]
#[variants(OutputT, f32, f16, bf16)]
#[variants(AccumT, f32, f16)]
pub fn qk_norm<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float,
    AccumT: ArrayElement + Float,
>(
    #[optional(!in_place)] qkv_input: Option<*const InputT>,
    scales: *const ScaleT,
    qkv_output: *mut OutputT,
    batch_size: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    epsilon: f32,
    scale_offset: f32,
    head_offset: u32,
    head_count: u32,
    full_layer: bool,
    #[specialize] in_place: bool,
) {
    let qkv_input = match in_place {
        true => qkv_output as *const InputT,
        false => qkv_input.unwrap(),
    };

    let head_dim = head_dim as usize;
    let head_offset = head_offset as usize;
    let head_count = head_count as usize;
    let qkv_stride = (num_q_heads + 2 * num_kv_heads) as usize * head_dim;
    let head_dim_accum = AccumT::from(head_dim).unwrap();
    let epsilon = AccumT::from(epsilon).unwrap();
    let scale_offset = AccumT::from(scale_offset).unwrap();

    for batch in 0..(batch_size as usize) {
        for head in 0..head_count {
            let offset = batch * qkv_stride + (head_offset + head) * head_dim;

            let mut total_sum = AccumT::zero();
            for i in 0..head_dim {
                let input_val = unsafe { AccumT::from(*qkv_input.add(offset + i)).unwrap() };
                total_sum = total_sum + input_val * input_val;
            }
            let mean_square: AccumT = total_sum / head_dim_accum;
            let rms_norm = AccumT::from((mean_square + epsilon).to_f32().unwrap().sqrt().recip()).unwrap();

            for i in 0..head_dim {
                let input_val = unsafe { AccumT::from(*qkv_input.add(offset + i)).unwrap() };
                let scale_val = unsafe { AccumT::from(*scales.add(i)).unwrap() };
                let normalized: AccumT = input_val * rms_norm;
                let result: OutputT = if full_layer {
                    let scale_with_offset: AccumT = scale_val + scale_offset;
                    OutputT::from(normalized * scale_with_offset).unwrap()
                } else {
                    let normalized_low = OutputT::from(normalized).unwrap();
                    let scale_value_low = OutputT::from(scale_val + scale_offset).unwrap();
                    normalized_low * scale_value_low
                };
                unsafe { *qkv_output.add(offset + i) = result };
            }
        }
    }
}
