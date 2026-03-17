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
    #[allow(unused)] input_scale: f32,
    #[allow(unused)]
    #[specialize]
    group_size: u32,
    #[allow(unused)]
    #[specialize]
    quant_mode: u32,
) {
    const QUANT_UINT4: u32 = 0;
    const QUANT_INT8: u32 = 1;

    let packing_divisor: u32 = if quant_mode == QUANT_UINT4 { 2 } else { 1 };
    let weights_stride = model_dim / packing_divisor;
    let num_groups = (model_dim + group_size - 1) / group_size;

    for batch_idx in 0..batch_size {
        let token_id = unsafe { *token_ids.add(batch_idx as usize) };
        for dim_idx in 0..model_dim {
            let output_idx = (batch_idx * model_dim + dim_idx) as usize;
            if token_id >= vocab_size as u64 {
                unsafe { *output.add(output_idx) = T::zero() };
                continue;
            }

            let group_idx = dim_idx / group_size;
            let scale_idx = (token_id as u32 * num_groups + group_idx) as usize;
            let scale = unsafe { *scales.add(scale_idx) };
            let bias = unsafe { *biases.add(scale_idx) };

            let quantized_value: i32 = unsafe {
                if quant_mode == QUANT_UINT4 {
                    let byte_idx = (token_id as u32 * weights_stride + dim_idx / 2) as usize;
                    let packed = *weights.add(byte_idx);
                    if (dim_idx & 1) == 0 {
                        (packed & 0x0F) as i32
                    } else {
                        ((packed >> 4) & 0x0F) as i32
                    }
                } else if quant_mode == QUANT_INT8 {
                    let elem_idx = (token_id as u32 * weights_stride + dim_idx) as usize;
                    *(weights as *const i8).add(elem_idx) as i32
                } else {
                    let elem_idx = (token_id as u32 * weights_stride + dim_idx) as usize;
                    *weights.add(elem_idx) as i32
                }
            };

            let out_f = scale.to_f32().unwrap() * quantized_value as f32
                + bias.to_f32().unwrap();
            let out_f = out_f * input_scale;
            unsafe { *output.add(output_idx) = T::from(out_f).unwrap() };
        }
    }
}
