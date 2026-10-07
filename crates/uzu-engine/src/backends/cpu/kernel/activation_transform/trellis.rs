use half::bf16;
use uzu_engine_macros::kernel;

use super::{hadamard_transform, min_max_symmetric_divisor, quantize_symmetric_i8};
use crate::backends::common::gpu_types::trellis::{COLUMN_GROUP_COUNT, mixing_dimension};

fn rotate_token(
    input: &[bf16],
    rht_factors: &[f32],
    mixing: &[f32],
    mixing_dimension: usize,
) -> Vec<f32> {
    let columns = input.len();
    let hadamard_size = columns / mixing_dimension;
    let mut rotated = vec![0.0f32; columns];
    for output_mixing_index in 0..mixing_dimension {
        let mut hadamard_values: Vec<f32> = (0..hadamard_size)
            .map(|hadamard_index| {
                (0..mixing_dimension).fold(0.0f32, |accumulated, mixing_index| {
                    let column = hadamard_index * mixing_dimension + mixing_index;
                    (input[column].to_f32() * rht_factors[column])
                        .mul_add(mixing[output_mixing_index * mixing_dimension + mixing_index], accumulated)
                })
            })
            .collect();
        hadamard_transform(&mut hadamard_values);
        for (hadamard_index, value) in hadamard_values.into_iter().enumerate() {
            rotated[hadamard_index * mixing_dimension + output_mixing_index] = bf16::from_f32(value).to_f32();
        }
    }
    rotated
}

#[kernel(TrellisTransform)]
#[variants(DIMENSION, 4096, 5120, 6144, 6656, 17408, 19968)]
pub fn trellis_transform<const DIMENSION: u32>(
    input: *const bf16,
    rht_factors: *const f32,
    mixing: *const f32,
    activations: *mut i8,
    column_group_sums: *mut f32,
    scales: *mut f32,
    batch: u32,
) {
    let columns = DIMENSION as usize;
    let mixing_dimension = mixing_dimension(DIMENSION) as usize;
    let rht_factors = unsafe { std::slice::from_raw_parts(rht_factors, columns) };
    let mixing = unsafe { std::slice::from_raw_parts(mixing, mixing_dimension * mixing_dimension) };

    for token in 0..batch as usize {
        let row = unsafe { std::slice::from_raw_parts(input.add(token * columns), columns) };
        let quantized_row = unsafe { std::slice::from_raw_parts_mut(activations.add(token * columns), columns) };
        let column_group_sums_row = unsafe {
            std::slice::from_raw_parts_mut(
                column_group_sums.add(COLUMN_GROUP_COUNT as usize * token),
                COLUMN_GROUP_COUNT as usize,
            )
        };

        let rotated = rotate_token(row, rht_factors, mixing, mixing_dimension);
        let scale = min_max_symmetric_divisor(&rotated);
        let mut token_column_group_sums = [0i32; COLUMN_GROUP_COUNT as usize];
        for (column, (quantized, &value)) in quantized_row.iter_mut().zip(&rotated).enumerate() {
            *quantized = quantize_symmetric_i8(value, scale);
            token_column_group_sums[column % COLUMN_GROUP_COUNT as usize] += i32::from(*quantized);
        }

        column_group_sums_row.copy_from_slice(&token_column_group_sums.map(|sum| sum as f32));
        unsafe { *scales.add(token) = scale };
    }
}
