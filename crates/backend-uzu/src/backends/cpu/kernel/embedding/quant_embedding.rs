use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{ArrayElement, backends::common::gpu_types::QuantizationMode};

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
    quantization_mode: crate::backends::common::gpu_types::quantization::QuantizationMode,
) {
    let packing_divisor = quantization_mode.packing_divisor() as u32;
    let weights_stride = model_dim / packing_divisor;
    let num_groups = (model_dim + group_size - 1) / group_size;

    unsafe {
        for batch_idx in 0..batch_size {
            let token_id = *token_ids.add(batch_idx as usize);

            for dim_idx in 0..model_dim {
                let out_idx = (batch_idx * model_dim + dim_idx) as usize;

                if token_id >= vocab_size as u64 {
                    *output.add(out_idx) = T::zero();
                    continue;
                }

                let group_idx = dim_idx / group_size;
                let scale = *scales.add((token_id as u32 * num_groups + group_idx) as usize);
                let bias = *biases.add((token_id as u32 * num_groups + group_idx) as usize);

                let quantized_value: i32 = match quantization_mode {
                    QuantizationMode::U4 => {
                        let byte_idx = (token_id as u32 * weights_stride + dim_idx / 2) as usize;
                        let packed = *weights.add(byte_idx);
                        if (dim_idx & 1) == 0 {
                            (packed & 0x0F) as i32
                        } else {
                            ((packed >> 4) & 0x0F) as i32
                        }
                    },
                    QuantizationMode::I8 => {
                        let elem_idx = (token_id as u32 * weights_stride + dim_idx) as usize;
                        let weights_i8 = weights as *const i8;
                        *weights_i8.add(elem_idx) as i32
                    },
                    QuantizationMode::U8 => {
                        let elem_idx = (token_id as u32 * weights_stride + dim_idx) as usize;
                        *weights.add(elem_idx) as i32
                    },
                };

                let out_f = scale.to_f32().unwrap() * quantized_value as f32 + bias.to_f32().unwrap();
                let out_f = out_f * input_scale;
                *output.add(out_idx) = T::from(out_f).unwrap();
            }
        }
    }
}
