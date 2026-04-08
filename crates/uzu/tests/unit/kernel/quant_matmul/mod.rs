#[macro_use]
#[path = "../../../common/mod.rs"]
mod common;

mod qmm_transposed_64x64_test;
mod qmm_transposed_test;
mod qmm_transposed_wide_test;
mod qmv_fast_test;
mod qmv_test;

use half::{bf16, f16};
use num_traits::Float;
use uzu::ArrayElement;

use crate::uzu_test;

pub(super) struct Input<T: ArrayElement + Float> {
    pub w_packed: Vec<u32>,
    pub scales: Vec<T>,
    pub zero_points: Option<Vec<u8>>,
    pub biases: Option<Vec<T>>,
    pub x: Vec<T>,
    pub k: u32,
    pub n: u32,
    pub m: u32,
    pub group_size: u32,
    pub bits: u32,
    pub use_zero_points: bool,
    pub use_mlx_quant: bool,
}

pub(super) fn pack_weights_u32(
    values: &[u8],
    bits: u32,
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
    bits: u32,
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

/// Validates the bit-manipulation used by uint_to_fp / uint4_to_fp4 in quant_matmul.h
/// for nibble values 0–15.
#[uzu_test]
fn test_uint4_to_fp_bit_trick() {
    // float path: as_type<float>(x | 0x4B000000) - 8388608.0
    for x in 0u32..=15 {
        let bits = x | 0x4B00_0000;
        let result = f32::from_bits(bits) - 8388608.0f32;
        assert_eq!(result, x as f32, "float path failed for x={x}");
    }

    // half path: narrows from float, exact for 0–15
    for x in 0u32..=15 {
        let bits = x | 0x4B00_0000;
        let via_float = f32::from_bits(bits) - 8388608.0f32;
        let as_half = f16::from_f32(via_float);
        assert_eq!(as_half.to_f32(), x as f32, "half path failed for x={x}");
    }

    // bfloat path: as_type<bfloat>((x as u16) | 0x4300) - bf16(128.0)
    for x in 0u32..=15 {
        let narrow = (x as u16) | 0x4300;
        let val = bf16::from_bits(narrow);
        let result = val.to_f32() - 128.0f32;
        assert_eq!(result, x as f32, "bfloat path failed for x={x}");
    }
}
