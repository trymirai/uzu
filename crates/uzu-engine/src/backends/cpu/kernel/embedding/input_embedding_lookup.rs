use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    array::ArrayElement,
    backends::{
        common::gpu_types::{EmbeddingTableKind, HADAMARD_TRANSFORM_BLOCK_SIZE, QuantizationMethod, QuantizationMode},
        cpu::kernel::activation_transform::hadamard_transform,
    },
};

#[kernel(InputEmbeddingLookup)]
#[variants(T, f32, bf16)]
pub fn input_embedding_lookup<T: ArrayElement + Float>(
    token_ids: *const u32,
    values: *const u8,
    #[optional(table_kind != EmbeddingTableKind::Dense)] scales: Option<*const T>,
    #[optional(quantization_method == QuantizationMethod::ScaleZeroPoint)] zero_points: Option<*const u8>,
    #[optional(quantization_method == QuantizationMethod::ScaleBias)] biases: Option<*const T>,
    #[optional(use_hadamard)] hadamard_factors: Option<*const i32>,
    #[optional(table_kind == EmbeddingTableKind::D4)] ladder_indices: Option<*const u8>,
    #[optional(table_kind == EmbeddingTableKind::D4)] ladder: Option<*const f16>,
    #[optional(table_kind == EmbeddingTableKind::D4)] codebook: Option<*const i8>,
    output: *mut T,
    batch_size: u32,
    vocab_size: u32,
    model_dim: u32,
    input_scale: f32,
    #[specialize] table_kind: EmbeddingTableKind,
    #[specialize] group_size: u32,
    #[specialize] quantization_mode: QuantizationMode,
    #[specialize] quantization_method: QuantizationMethod,
    #[specialize] use_hadamard: bool,
) {
    let factors = match table_kind {
        EmbeddingTableKind::Dense => None,
        EmbeddingTableKind::Quantized => hadamard_factors.filter(|_| use_hadamard),
        EmbeddingTableKind::D4 => Some(hadamard_factors.expect("D4 lookup requires Hadamard factors")),
    };
    let dim = model_dim as usize;
    let (num_groups, weights_stride) = if table_kind == EmbeddingTableKind::Quantized {
        (dim.div_ceil(group_size as usize), dim / quantization_mode.packing_divisor() as usize)
    } else {
        (0, 0)
    };
    let dense_scale = T::from(input_scale).unwrap();
    for batch in 0..batch_size as usize {
        let row = unsafe { std::slice::from_raw_parts_mut(output.add(batch * dim), dim) };
        let token = unsafe { *token_ids.add(batch) };
        if token >= vocab_size {
            row.fill(T::zero());
            continue;
        }
        let token = token as usize;
        let load = |column: usize| -> f32 {
            match table_kind {
                EmbeddingTableKind::Dense => unsafe {
                    (*(values as *const T).add(token * dim + column) * dense_scale).to_f32().unwrap()
                },
                EmbeddingTableKind::Quantized => {
                    let scales = scales.expect("quantized lookup requires scales");
                    let group = column / group_size as usize;
                    let index = token * num_groups + group;
                    let scale = unsafe { (*scales.add(index)).to_f32().unwrap() };
                    let offset = token * weights_stride;
                    let code = match quantization_mode {
                        QuantizationMode::U4 => read_u4(values, 2 * offset + column) as f32,
                        QuantizationMode::I8 => (unsafe { *(values as *const i8).add(offset + column) }) as f32,
                        QuantizationMode::U8 => (unsafe { *values.add(offset + column) }) as f32,
                    };
                    let bias = match quantization_method {
                        QuantizationMethod::ScaleBias => unsafe {
                            (*biases.expect("quantized lookup requires biases").add(index)).to_f32().unwrap()
                        },
                        QuantizationMethod::ScaleZeroPoint => {
                            let zero_points = zero_points.expect("quantized lookup requires zero points");
                            let zero_point = if quantization_mode == QuantizationMode::U4 {
                                read_u4(zero_points, 2 * token * num_groups.div_ceil(2) + group)
                            } else {
                                unsafe { *zero_points.add(index) }
                            };
                            -scale * zero_point as f32
                        },
                        QuantizationMethod::ScaleSymmetric => {
                            let midpoint = if quantization_mode == QuantizationMode::U4 {
                                8.0
                            } else {
                                128.0
                            };
                            -scale * midpoint
                        },
                    };
                    // Ordinary quantization rounds to T before the transform.
                    T::from((scale * code + bias) * input_scale).unwrap().to_f32().unwrap()
                },
                EmbeddingTableKind::D4 => {
                    let row_scales = scales.expect("D4 lookup requires row scales");
                    let ladder_indices = ladder_indices.expect("D4 lookup requires ladder indices");
                    let ladder = ladder.expect("D4 lookup requires a ladder");
                    let codebook = codebook.expect("D4 lookup requires a codebook");
                    let ladder_index = read_u4(ladder_indices, 2 * token * (dim / 128) + column / 64);
                    let code = unsafe { *values.add(token * (dim / 4) + column / 4) } as usize;
                    let point = unsafe { *codebook.add(4 * code + column % 4) };
                    let row_scale = unsafe { (*row_scales.add(token)).to_f32().unwrap() };
                    let step = unsafe { (*ladder.add(ladder_index as usize)).to_f32() };
                    row_scale * step * point as f32 * input_scale
                },
            }
        };
        if let Some(factors) = factors {
            for block_start in (0..row.len()).step_by(HADAMARD_TRANSFORM_BLOCK_SIZE as usize) {
                let mut block: [f32; HADAMARD_TRANSFORM_BLOCK_SIZE as usize] =
                    std::array::from_fn(|lane| load(block_start + lane));
                hadamard_transform(&mut block);
                for (lane, value) in block.into_iter().enumerate() {
                    row[block_start + lane] =
                        T::from(value * unsafe { *factors.add(block_start + lane) } as f32).unwrap();
                }
            }
        } else {
            for (column, value) in row.iter_mut().enumerate() {
                *value = T::from(load(column)).unwrap();
            }
        }
    }
}

fn read_u4(
    values: *const u8,
    nibble: usize,
) -> u8 {
    (unsafe { *values.add(nibble / 2) } >> (4 * (nibble % 2))) & 15
}
