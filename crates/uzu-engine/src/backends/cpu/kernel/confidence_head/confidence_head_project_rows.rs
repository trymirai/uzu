use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use super::math::{dense, layer_norm};
use crate::array::ArrayElement;

/// dense(layer_norm(row)) of every row, in f32: the drafter rows of a step (outputs in f32) and, at load, the
/// projected token table (outputs in bf16).
#[kernel(ConfidenceHeadProjectRows)]
#[variants(T, f32, bf16)]
pub fn confidence_head_project_rows<T: ArrayElement + Float>(
    rows: *const T,
    norm_scales: *const f32,
    norm_biases: *const f32,
    weights: *const f32,
    biases: *const f32,
    #[optional(!output_bf16)] output: Option<*mut f32>,
    #[optional(output_bf16)] output_half: Option<*mut bf16>,
    row_count: u32,
    epsilon: f32,
    #[specialize] output_bf16: bool,
    #[specialize] width: u32,
    #[specialize] outputs: u32,
) {
    let (width, outputs) = (width as usize, outputs as usize);
    let scales = unsafe { std::slice::from_raw_parts(norm_scales, width) };
    let shifts = unsafe { std::slice::from_raw_parts(norm_biases, width) };
    let weights = unsafe { std::slice::from_raw_parts(weights, outputs * width) };
    let biases = unsafe { std::slice::from_raw_parts(biases, outputs) };
    for row in 0..row_count as usize {
        let values =
            (0..width).map(|index| unsafe { (*rows.add(row * width + index)).to_f32().unwrap() }).collect::<Vec<_>>();
        let projected = dense(weights, biases, &layer_norm(&values, scales, shifts, epsilon));
        for (index, value) in projected.into_iter().enumerate() {
            unsafe {
                match output_bf16 {
                    true => *output_half.unwrap().add(row * outputs + index) = bf16::from_f32(value),
                    false => *output.unwrap().add(row * outputs + index) = value,
                }
            }
        }
    }
}
