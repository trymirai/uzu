mod qmm_test;
mod qmm_transposed_64x64_test;
mod qmm_transposed_test;
mod qmm_transposed_wide_test;

use num_traits::Float;
use uzu::ArrayElement;

pub(super) struct Input<T: ArrayElement + Float> {
    pub w_packed: Vec<u32>,
    pub scales: Vec<T>,
    pub zero_points: Option<Vec<u8>>,
    pub biases: Option<Vec<T>>,
    pub x: Vec<T>,
    pub k: i32,
    pub n: i32,
    pub m: i32,
    pub group_size: i32,
    pub bits: i32,
    pub use_zero_points: bool,
    pub use_mlx_quant: bool,
}

pub(super) fn pack_weights_u32(
    values: &[u8],
    bits: i32,
) -> Vec<u32> {
    if bits == 4 {
        values
            .chunks(8)
            .map(|chunk| {
                let mut word = 0u32;
                for (i, &v) in chunk.iter().enumerate() {
                    word |= ((v & 0xF) as u32) << (i * 4);
                }
                word
            })
            .collect()
    } else {
        values
            .chunks(4)
            .map(|chunk| {
                let mut word = 0u32;
                for (i, &v) in chunk.iter().enumerate() {
                    word |= (v as u32) << (i * 8);
                }
                word
            })
            .collect()
    }
}

pub(super) fn pack_zero_points(
    values: &[u8],
    bits: i32,
) -> Vec<u8> {
    if bits == 4 {
        values
            .chunks(2)
            .map(|chunk| {
                let lo = chunk[0] & 0x0F;
                let hi = if chunk.len() > 1 {
                    chunk[1] & 0x0F
                } else {
                    0
                };
                lo | (hi << 4)
            })
            .collect()
    } else {
        values.to_vec()
    }
}

pub(super) fn check_tolerance(
    expected: f32,
    actual: f32,
    rel_tol: f64,
    abs_tol: f64,
) -> bool {
    let diff = (expected - actual).abs() as f64;
    let tol = abs_tol.max(expected.abs() as f64 * rel_tol);
    diff <= tol
}
