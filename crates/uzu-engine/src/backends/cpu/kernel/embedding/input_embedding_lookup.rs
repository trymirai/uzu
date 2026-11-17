use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    backends::{
        common::gpu_types::{
            d4s4, embedding::EmbeddingTableKind, quantization::QuantizationMode,
            quantization_method::QuantizationMethod,
        },
        cpu::kernel::activation_transform::hadamard_transform,
    },
    data_type::DataTypeRepr,
};

#[kernel(InputEmbeddingLookup)]
#[variants(T, f32, bf16)]
pub fn input_embedding_lookup<T: DataTypeRepr + Float>(
    token_ids: *const u32,
    values: *const u8,
    #[optional(table_kind != EmbeddingTableKind::Dense)] scales: Option<*const T>,
    #[optional(quantization_method == QuantizationMethod::ScaleZeroPoint)] zero_points: Option<*const u8>,
    #[optional(quantization_method == QuantizationMethod::ScaleBias)] biases: Option<*const T>,
    #[optional(hadamard_size != 0)] hadamard_factors: Option<*const i32>,
    #[optional(table_kind == EmbeddingTableKind::D4S4)] ladder_indices: Option<*const u8>,
    #[optional(table_kind == EmbeddingTableKind::D4S4)] ladder: Option<*const f16>,
    #[optional(table_kind == EmbeddingTableKind::D4S4)] codebook: Option<*const i8>,
    output: *mut T,
    batch_size: u32,
    vocab_size: u32,
    first_row: u32,
    model_dim: u32,
    input_scale: f32,
    #[specialize] table_kind: EmbeddingTableKind,
    #[specialize]
    #[optional(table_kind == EmbeddingTableKind::Quantized)]
    group_size: Option<u32>,
    #[specialize]
    #[optional(table_kind == EmbeddingTableKind::Quantized)]
    quantization_mode: Option<QuantizationMode>,
    #[specialize]
    #[optional(table_kind == EmbeddingTableKind::Quantized)]
    quantization_method: Option<QuantizationMethod>,
    #[specialize] hadamard_size: u32,
) {
    let hadamard_signs = if hadamard_size != 0 {
        hadamard_factors
    } else {
        None
    };
    let dim = model_dim as usize;
    let quantization =
        if let (Some(group_size), Some(mode), Some(method)) = (group_size, quantization_mode, quantization_method) {
            let num_groups = dim.div_ceil(group_size as usize);
            let weights_stride = dim / mode.packing_divisor() as usize;
            Some((group_size, mode, method, num_groups, weights_stride))
        } else {
            None
        };
    let dense_scale = T::from(input_scale).unwrap();
    for batch in 0..batch_size as usize {
        let output_row = unsafe { std::slice::from_raw_parts_mut(output.add(batch * dim), dim) };
        let Some(token) = unsafe { *token_ids.add(batch) }.checked_sub(first_row) else {
            continue;
        };
        if token >= vocab_size {
            output_row.fill(T::zero());
            continue;
        }
        let token = token as usize;
        let decode_weight = |column: usize| -> f32 {
            match table_kind {
                EmbeddingTableKind::Dense => unsafe {
                    (*(values as *const T).add(token * dim + column) * dense_scale).to_f32().unwrap()
                },
                EmbeddingTableKind::Quantized => {
                    let (group_size, quantization_mode, quantization_method, num_groups, weights_stride) =
                        quantization.expect("quantization settings");
                    let scales = scales.expect("quantized lookup requires scales");
                    let group = column / group_size as usize;
                    let index = token * num_groups + group;
                    let scale = unsafe { (*scales.add(index)).to_f32().unwrap() };
                    let offset = token * weights_stride;
                    let code = match quantization_mode {
                        QuantizationMode::U2 => read_packed(values, 4 * offset + column, 2) as f32,
                        QuantizationMode::U4 => read_packed(values, 2 * offset + column, 4) as f32,
                        QuantizationMode::I8 => (unsafe { *(values as *const i8).add(offset + column) }) as f32,
                        QuantizationMode::U8 => (unsafe { *values.add(offset + column) }) as f32,
                    };
                    let bias = match quantization_method {
                        QuantizationMethod::ScaleBias => unsafe {
                            (*biases.expect("quantized lookup requires biases").add(index)).to_f32().unwrap()
                        },
                        QuantizationMethod::ScaleZeroPoint => {
                            let zero_points = zero_points.expect("quantized lookup requires zero points");
                            let packing = quantization_mode.packing_divisor() as usize;
                            let zero_point = if packing > 1 {
                                read_packed(
                                    zero_points,
                                    packing * token * num_groups.div_ceil(packing) + group,
                                    8 / packing,
                                )
                            } else {
                                unsafe { *zero_points.add(index) }
                            };
                            -scale * zero_point as f32
                        },
                        QuantizationMethod::ScaleSymmetric => {
                            let midpoint = (1u32 << (8 / quantization_mode.packing_divisor() - 1)) as f32;
                            -scale * midpoint
                        },
                    };
                    // Ordinary quantization rounds to T before the transform.
                    T::from((scale * code + bias) * input_scale).unwrap().to_f32().unwrap()
                },
                EmbeddingTableKind::D4S4 => {
                    let row_scales = scales.expect("D4S4 lookup requires row scales");
                    let ladder_indices = ladder_indices.expect("D4S4 lookup requires ladder indices");
                    let ladder = ladder.expect("D4S4 lookup requires a ladder");
                    let codebook = codebook.expect("D4S4 lookup requires a codebook");
                    let values_per_code = d4s4::VALUES_PER_CODE as usize;
                    let columns_per_scale = d4s4::COLUMNS_PER_LADDER_SCALE as usize;
                    let ladder_index =
                        read_packed(ladder_indices, token * (dim / columns_per_scale) + column / columns_per_scale, 4);
                    let code =
                        unsafe { *values.add(token * (dim / values_per_code) + column / values_per_code) } as usize;
                    let point = unsafe { *codebook.add(values_per_code * code + column % values_per_code) };
                    let row_scale = unsafe { (*row_scales.add(token)).to_f32().unwrap() };
                    let step = unsafe { (*ladder.add(ladder_index as usize)).to_f32() };
                    row_scale * step * point as f32 * input_scale
                },
            }
        };
        if let Some(signs) = hadamard_signs {
            let block_size = hadamard_size as usize;
            let mut scratch = [0.0f32; 128];
            let decoded_block = &mut scratch[..block_size];
            for (block_index, output_block) in output_row.chunks_mut(block_size).enumerate() {
                let first_column = block_index * block_size;
                for (column_in_block, decoded_value) in decoded_block.iter_mut().enumerate() {
                    *decoded_value = decode_weight(first_column + column_in_block);
                }
                hadamard_transform(decoded_block);
                for (column_in_block, output_value) in output_block.iter_mut().enumerate() {
                    let sign = unsafe { *signs.add(first_column + column_in_block) } as f32;
                    *output_value = T::from(decoded_block[column_in_block] * sign).unwrap();
                }
            }
        } else {
            for (column, value) in output_row.iter_mut().enumerate() {
                *value = T::from(decode_weight(column)).unwrap();
            }
        }
    }
}

fn read_packed(
    values: *const u8,
    index: usize,
    bits: usize,
) -> u8 {
    (unsafe { *values.add(index / (8 / bits)) } >> (bits * (index % (8 / bits)))) & ((1 << bits) - 1)
}
