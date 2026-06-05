use half::{bf16, f16};
use num_traits::{Float, NumCast};
use proc_macros::kernel;

use crate::{array::ArrayElement, backends::common::gpu_types::quantization_method::QuantizationMethod};

#[kernel(QuantizedMatmulQmvFastLloydMaxMerged)]
#[variants(T, f32, bf16)]
#[variants(GROUP_SIZE, 32, 64, 128)]
#[variants(BITS, 4)]
pub fn quantized_matmul_qmv_fast_lloyd_max_merged<T: ArrayElement + Float, const GROUP_SIZE: u32, const BITS: u32>(
    weights: *const u32,
    scales: *const T,
    #[optional(quant_method == QuantizationMethod::ScaleZeroPoint)] zero_points: Option<*const u8>,
    #[optional(quant_method == QuantizationMethod::ScaleBias)] biases: Option<*const T>,
    #[optional(quant_method == QuantizationMethod::LloydMax)] codebook: Option<*const f16>,
    #[optional(quant_method == QuantizationMethod::LloydMax)] bias_indices: Option<*const u8>,
    #[optional(quant_method == QuantizationMethod::LloydMax)] bias_codebook: Option<*const f16>,
    input: *const T,
    output: *mut T,
    in_vec_size: u32,
    out_vec_size: u32,
    batch_size: u32,
    #[allow(non_snake_case)]
    #[specialize]
    quant_method: QuantizationMethod,
) {
    let _ = BITS;
    let in_vec_size = in_vec_size as usize;
    let out_vec_size = out_vec_size as usize;
    let batch_size = batch_size as usize;

    match quant_method {
        QuantizationMethod::LloydMax => {
            let (Some(codebook), Some(bias_indices), Some(bias_codebook)) = (codebook, bias_indices, bias_codebook)
            else {
                return;
            };
            quantized_matmul_qmv_lloyd_max::<T, GROUP_SIZE>(
                weights,
                scales,
                codebook,
                bias_indices,
                bias_codebook,
                input,
                output,
                in_vec_size,
                out_vec_size,
                batch_size,
            );
        },
        QuantizationMethod::ScaleBias => quantized_matmul_qmv_affine::<T, GROUP_SIZE>(
            weights,
            scales,
            None,
            biases,
            input,
            output,
            in_vec_size,
            out_vec_size,
            batch_size,
        ),
        QuantizationMethod::ScaleZeroPoint => quantized_matmul_qmv_affine::<T, GROUP_SIZE>(
            weights,
            scales,
            zero_points,
            None,
            input,
            output,
            in_vec_size,
            out_vec_size,
            batch_size,
        ),
        QuantizationMethod::ScaleSymmetric => quantized_matmul_qmv_affine::<T, GROUP_SIZE>(
            weights,
            scales,
            None,
            None,
            input,
            output,
            in_vec_size,
            out_vec_size,
            batch_size,
        ),
    }
}

pub fn quantized_matmul_qmv_lloyd_max<T: ArrayElement + Float, const GROUP_SIZE: u32>(
    weights: *const u32,
    scales: *const T,
    codebook: *const f16,
    bias_indices: *const u8,
    bias_codebook: *const f16,
    input: *const T,
    output: *mut T,
    in_vec_size: usize,
    out_vec_size: usize,
    batch_size: usize,
) {
    let groups_per_row = in_vec_size / GROUP_SIZE as usize;
    let packed_words_per_row = in_vec_size / 8;
    let bias_stride = groups_per_row.div_ceil(2);

    for batch_index in 0..batch_size {
        for output_index in 0..out_vec_size {
            let mut accumulator = 0.0f32;
            for input_index in 0..in_vec_size {
                let weight_code = read_u4(weights, output_index * packed_words_per_row, input_index) as usize;
                let group_index = input_index / GROUP_SIZE as usize;
                let bias_byte = unsafe { *bias_indices.add(output_index * bias_stride + (group_index >> 1)) };
                let bias_code = if (group_index & 1) == 0 {
                    bias_byte & 0x0f
                } else {
                    (bias_byte >> 4) & 0x0f
                } as usize;
                let scale = unsafe { (*scales.add(output_index * groups_per_row + group_index)).to_f32().unwrap() };
                let codebook_value = unsafe { (*codebook.add(weight_code)).to_f32() };
                let bias_value = unsafe { (*bias_codebook.add(bias_code)).to_f32() };
                let input_value = unsafe { (*input.add(batch_index * in_vec_size + input_index)).to_f32().unwrap() };
                accumulator += input_value * scale * (codebook_value - bias_value);
            }
            unsafe {
                *output.add(batch_index * out_vec_size + output_index) = NumCast::from(accumulator).unwrap();
            }
        }
    }
}

fn quantized_matmul_qmv_affine<T: ArrayElement + Float, const GROUP_SIZE: u32>(
    weights: *const u32,
    scales: *const T,
    zero_points: Option<*const u8>,
    biases: Option<*const T>,
    input: *const T,
    output: *mut T,
    in_vec_size: usize,
    out_vec_size: usize,
    batch_size: usize,
) {
    let groups_per_row = in_vec_size.div_ceil(GROUP_SIZE as usize);
    let packed_words_per_row = in_vec_size.div_ceil(8);
    let zero_point_stride = groups_per_row.div_ceil(2);

    for batch_index in 0..batch_size {
        for output_index in 0..out_vec_size {
            let mut accumulator = 0.0f32;
            for input_index in 0..in_vec_size {
                let group_index = input_index / GROUP_SIZE as usize;
                let quantized_value = read_u4(weights, output_index * packed_words_per_row, input_index) as f32;
                let scale = unsafe { (*scales.add(output_index * groups_per_row + group_index)).to_f32().unwrap() };
                let bias = if let Some(zero_points) = zero_points {
                    -scale * read_packed_u4(zero_points, output_index * zero_point_stride, group_index) as f32
                } else if let Some(biases) = biases {
                    unsafe { (*biases.add(output_index * groups_per_row + group_index)).to_f32().unwrap() }
                } else {
                    -scale * 8.0
                };
                let input_value = unsafe { (*input.add(batch_index * in_vec_size + input_index)).to_f32().unwrap() };
                accumulator += input_value * (scale * quantized_value + bias);
            }
            unsafe {
                *output.add(batch_index * out_vec_size + output_index) = NumCast::from(accumulator).unwrap();
            }
        }
    }
}

fn read_u4(
    words: *const u32,
    row_word_offset: usize,
    element_index: usize,
) -> u8 {
    let word_index = row_word_offset + element_index / 8;
    let bit_offset = (element_index % 8) * 4;
    unsafe { ((*words.add(word_index) >> bit_offset) & 0x0f) as u8 }
}

fn read_packed_u4(
    values: *const u8,
    row_byte_offset: usize,
    element_index: usize,
) -> u8 {
    let byte = unsafe { *values.add(row_byte_offset + (element_index >> 1)) };
    if (element_index & 1) == 0 {
        byte & 0x0f
    } else {
        (byte >> 4) & 0x0f
    }
}
