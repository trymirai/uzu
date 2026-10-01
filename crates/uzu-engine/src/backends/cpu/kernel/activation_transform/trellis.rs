use half::bf16;
use uzu_engine_macros::kernel;

use super::{min_max_symmetric_divisor, quantize_symmetric_i8};
use crate::backends::common::{gpu_types::trellis, kernel::mixing_order};

const COLUMN_CLASS_COUNT: usize = trellis::COLUMN_CLASS_COUNT as usize;
const TOKEN_STATISTICS_LEN: usize = trellis::TOKEN_STATISTICS_LEN as usize;
const SHORT_HADAMARD_SIZE: usize = trellis::SHORT_HADAMARD_SIZE as usize;
const LONG_HADAMARD_SIZE: usize = trellis::LONG_HADAMARD_SIZE as usize;

fn rotate_token(
    input: &[bf16],
    signs: &[f32],
    mixing: &[f32],
    mixing_order: usize,
) -> Vec<f32> {
    let columns = input.len();
    let hadamard_size = columns / mixing_order;
    let normalization = if hadamard_size == LONG_HADAMARD_SIZE {
        std::f32::consts::FRAC_1_SQRT_2 / (SHORT_HADAMARD_SIZE as f32).sqrt()
    } else {
        1.0 / (SHORT_HADAMARD_SIZE as f32).sqrt()
    };
    let mut rotated = vec![0.0f32; columns];
    for output_mixing_index in 0..mixing_order {
        let mut hadamard_values: Vec<f32> = (0..hadamard_size)
            .map(|hadamard_index| {
                (0..mixing_order).fold(0.0f32, |accumulated, mixing_index| {
                    let column = hadamard_index * mixing_order + mixing_index;
                    (input[column].to_f32() * signs[column])
                        .mul_add(mixing[output_mixing_index * mixing_order + mixing_index], accumulated)
                })
            })
            .collect();
        let mut stride = 1;
        while stride < hadamard_size {
            for lower_index in (0..hadamard_size).filter(|index| index & stride == 0) {
                let (lower, upper) = (hadamard_values[lower_index], hadamard_values[lower_index + stride]);
                hadamard_values[lower_index] = lower + upper;
                hadamard_values[lower_index + stride] = lower - upper;
            }
            stride <<= 1;
        }
        for (hadamard_index, value) in hadamard_values.into_iter().enumerate() {
            rotated[hadamard_index * mixing_order + output_mixing_index] =
                bf16::from_f32(value * normalization).to_f32();
        }
    }
    rotated
}

#[kernel(TrellisTransform)]
#[variants(DIMENSION, 5120, 6144, 17408)]
pub fn trellis_transform<const DIMENSION: u32>(
    input: *const bf16,
    signs: *const f32,
    mixing: *const f32,
    activations: *mut i8,
    token_statistics: *mut f32,
    batch: u32,
) {
    let columns = DIMENSION as usize;
    let mixing_order = mixing_order(DIMENSION) as usize;
    let signs = unsafe { std::slice::from_raw_parts(signs, columns) };
    let mixing = unsafe { std::slice::from_raw_parts(mixing, mixing_order * mixing_order) };

    for token in 0..batch as usize {
        let row = unsafe { std::slice::from_raw_parts(input.add(token * columns), columns) };
        let quantized_row = unsafe { std::slice::from_raw_parts_mut(activations.add(token * columns), columns) };
        let statistics = unsafe {
            std::slice::from_raw_parts_mut(token_statistics.add(TOKEN_STATISTICS_LEN * token), TOKEN_STATISTICS_LEN)
        };

        let rotated = rotate_token(row, signs, mixing, mixing_order);
        let scale = min_max_symmetric_divisor(&rotated);
        let mut class_sums = [0i32; COLUMN_CLASS_COUNT];
        for (column, (quantized, &value)) in quantized_row.iter_mut().zip(&rotated).enumerate() {
            *quantized = quantize_symmetric_i8(value, scale);
            class_sums[column % COLUMN_CLASS_COUNT] += i32::from(*quantized);
        }

        let (class_sums_out, scale_out) = statistics.split_at_mut(COLUMN_CLASS_COUNT);
        class_sums_out.copy_from_slice(&class_sums.map(|sum| sum as f32));
        scale_out[0] = scale;
        scale_out[1..].fill(0.0);
    }
}
