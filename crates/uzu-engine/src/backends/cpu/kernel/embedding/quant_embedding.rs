use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    array::ArrayElement,
    backends::{
        common::gpu_types::{HADAMARD_TRANSFORM_BLOCK_SIZE, QuantizationMethod, QuantizationMode},
        cpu::kernel::activation_transform::hadamard_transform,
    },
    data_type::DataType,
};

#[kernel(QuantizedEmbeddingLookup)]
#[variants(T, f32, f16, bf16)]
pub fn quantized_embedding_lookup<T: ArrayElement + Float>(
    token_ids: *const u32,
    weights: *const u8,
    scales: *const T,
    #[optional(quantization_method == QuantizationMethod::ScaleZeroPoint)] zero_points: Option<*const u8>,
    #[optional(quantization_method == QuantizationMethod::ScaleBias)] biases: Option<*const T>,
    output: *mut T,
    #[optional(use_hadamard)] output_hadamard_factors: Option<*const i32>,
    batch_size: u32,
    vocab_size: u32,
    model_dim: u32,
    input_scale: f32,
    #[specialize] group_size: u32,
    #[specialize] quantization_mode: QuantizationMode,
    #[specialize] quantization_method: QuantizationMethod,
    #[specialize] use_hadamard: bool,
) {
    assert_eq!(output_hadamard_factors.is_some(), use_hadamard);
    assert!(!use_hadamard || model_dim.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE));

    let packing_divisor = quantization_mode.packing_divisor();
    let weights_stride = model_dim / packing_divisor;
    let num_groups = model_dim.div_ceil(group_size);
    let zero_points_stride = match quantization_mode {
        QuantizationMode::U4 => num_groups.div_ceil(2),
        QuantizationMode::I8 | QuantizationMode::U8 => num_groups,
    };

    unsafe {
        for batch_idx in 0..batch_size {
            let token_id = *token_ids.add(batch_idx as usize);

            for dim_idx in 0..model_dim {
                let out_idx = (batch_idx * model_dim + dim_idx) as usize;

                if token_id >= vocab_size {
                    *output.add(out_idx) = T::zero();
                    continue;
                }

                let group_idx = dim_idx / group_size;
                let scale = *scales.add((token_id * num_groups + group_idx) as usize);

                let quantized_value: i32 = match quantization_mode {
                    QuantizationMode::U4 => {
                        let byte_idx = (token_id * weights_stride + dim_idx / 2) as usize;
                        let packed = *weights.add(byte_idx);
                        if (dim_idx & 1) == 0 {
                            (packed & 0x0F) as i32
                        } else {
                            ((packed >> 4) & 0x0F) as i32
                        }
                    },
                    QuantizationMode::I8 => {
                        let elem_idx = (token_id * weights_stride + dim_idx) as usize;
                        let weights_i8 = weights as *const i8;
                        *weights_i8.add(elem_idx) as i32
                    },
                    QuantizationMode::U8 => {
                        let elem_idx = (token_id * weights_stride + dim_idx) as usize;
                        *weights.add(elem_idx) as i32
                    },
                };

                let bias = match quantization_method {
                    QuantizationMethod::ScaleBias => biases
                        .expect("ScaleBias quantized embedding requires biases")
                        .add((token_id * num_groups + group_idx) as usize)
                        .read()
                        .to_f32()
                        .unwrap(),
                    QuantizationMethod::ScaleZeroPoint => {
                        let zero_points = zero_points.expect("ScaleZeroPoint quantized embedding requires zero_points");
                        let zero_point = match quantization_mode {
                            QuantizationMode::U4 => {
                                let byte_idx = (token_id * zero_points_stride + group_idx / 2) as usize;
                                let packed = *zero_points.add(byte_idx);
                                if (group_idx & 1) == 0 {
                                    packed & 0x0F
                                } else {
                                    (packed >> 4) & 0x0F
                                }
                            },
                            QuantizationMode::I8 | QuantizationMode::U8 => {
                                *zero_points.add((token_id * zero_points_stride + group_idx) as usize)
                            },
                        };
                        -scale.to_f32().unwrap() * zero_point as f32
                    },
                    QuantizationMethod::ScaleSymmetric => {
                        let midpoint = 1 << (DataType::from(quantization_mode).size_in_bits() - 1);
                        -scale.to_f32().unwrap() * midpoint as f32
                    },
                };

                let out_f = scale.to_f32().unwrap() * quantized_value as f32 + bias;
                let out_f = out_f * input_scale;
                *output.add(out_idx) = T::from(out_f).unwrap();
            }

            // Output RHT like the Metal kernel: round to T, transform each block, then apply the factors
            if let Some(factors) = output_hadamard_factors {
                let row =
                    std::slice::from_raw_parts_mut(output.add((batch_idx * model_dim) as usize), model_dim as usize);
                for block_start in (0..model_dim as usize).step_by(HADAMARD_TRANSFORM_BLOCK_SIZE as usize) {
                    let mut block = std::array::from_fn(|lane| row[block_start + lane].to_f32().unwrap());
                    hadamard_transform(&mut block);
                    for (lane, value) in block.into_iter().enumerate() {
                        let factor = *factors.add(block_start + lane) as f32;
                        row[block_start + lane] = T::from(value * factor).unwrap();
                    }
                }
            }
        }
    }
}
