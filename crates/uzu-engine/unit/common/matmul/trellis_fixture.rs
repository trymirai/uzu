//! The fixture both trellis kernel tests are written against.
//!
//! One tape, one set of int8 codes derived from it by the host oracle, and one
//! f64 reference product over those codes — so a rounding difference in the
//! weight decode can never be mistaken for a matmul bug. The GEMM and the GEMV
//! differ only in the activation they feed in, which is what [`oracle`] takes as
//! a closure.
//!
//! THE BIT-EXACT PROBE. One-hot activations with unit row scales make each
//! output exactly one decoded weight, recovered as an integer and compared
//! against the code the oracle's *recurrence* produces — not its window read.
//! That pins the kernel's window read and the tape packing against the trellis
//! definition rather than against another window read. [`probe_points`] covers
//! both ends of a row, every coordinate of a step, and every alignment of the
//! two-word window read.

use half::bf16;

use crate::{
    backends::{
        common::{
            Allocation,
            kernel::matmul::{
                MatmulB,
                trellis_format::{TrellisConfig, TrellisTape, codebook_scale, hash_params, state_codes},
            },
        },
        metal::{Metal, MetalContext},
    },
    tests::helpers::alloc_allocation_with_data,
};

/// `L = 32` is the widest window, so the state mask is the one that exercises
/// the two-word read at every shift.
pub const CONFIG: TrellisConfig = TrellisConfig::new(32, 3);

pub struct Fixture {
    pub tape: TrellisTape,
    /// `[n][k]` int8 codes — the weights in the basis the MXU sees, before
    /// `row_scale * codebook_scale()`.
    pub codes: Vec<i8>,
    pub row_scales: Vec<bf16>,
    tape_buffer: Allocation<Metal>,
    scales_buffer: Allocation<Metal>,
}

impl Fixture {
    pub fn new(
        context: &MetalContext,
        config: TrellisConfig,
        n: u32,
        k: u32,
        row_scales: Vec<bf16>,
    ) -> Self {
        let tape = TrellisTape::random(config, n, k, 0x1234_5EED);
        Self {
            codes: tape.codes(),
            tape_buffer: alloc_allocation_with_data::<Metal, u32>(context, &tape.words),
            scales_buffer: alloc_allocation_with_data::<Metal, bf16>(context, &row_scales),
            tape,
            row_scales,
        }
    }

    pub fn weights(&self) -> MatmulB<'_, Metal> {
        MatmulB::Trellis {
            b: &self.tape_buffer,
            scales: &self.scales_buffer,
            config: self.tape.config,
        }
    }
}

/// Per-row weight scales, deterministic and all distinct.
pub fn row_scales(n: u32) -> Vec<bf16> {
    (0..n).map(|r| bf16::from_f32(0.002 + 0.001 * ((r * 37 % 19) as f32) / 19.0)).collect()
}

/// `[m][n]` in f64 over the same int8 codes the GPU decoded. `activation` is the
/// value the kernel actually consumed at `(row, k index)`, so activation
/// rounding is not in the result.
pub fn oracle(
    fixture: &Fixture,
    m: u32,
    n: u32,
    k: u32,
    activation: impl Fn(usize, usize) -> f64,
) -> Vec<f32> {
    let scale = f64::from(codebook_scale());
    let mut out = vec![0.0f32; (m * n) as usize];
    for row in 0..m as usize {
        for column in 0..n as usize {
            let mut accumulated = 0.0f64;
            for index in 0..k as usize {
                accumulated += activation(row, index) * f64::from(fixture.codes[column * k as usize + index]);
            }
            out[row * n as usize + column] =
                (accumulated * f64::from(fixture.row_scales[column].to_f32()) * scale) as f32;
        }
    }
    out
}

pub fn max_relative_error(
    got: &[bf16],
    expected: &[f32],
) -> f32 {
    let scale = expected.iter().fold(0.0f32, |acc, v| acc.max(v.abs())).max(f32::MIN_POSITIVE);
    got.iter().zip(expected).fold(0.0f32, |acc, (&g, &e)| acc.max((g.to_f32() - e).abs())) / scale
}

/// `(step, coordinate)` pairs for the bit-exact probe.
pub fn probe_points(steps: u32) -> Vec<(u32, u32)> {
    [0, 1, 2, 3, 5, 33, steps / 2, steps - 1].iter().flat_map(|&s| (0..4u32).map(move |j| (s, j))).collect()
}

/// One probe row: `got` is the kernel's `[n]` output for a one-hot activation at
/// `(step, coordinate)` with unit row scales, so every element is exactly one
/// decoded level times `codebook_scale()`.
///
/// Dividing that scale back out is exact enough to compare as an integer: the
/// levels are in `[-54, 55]`, and bf16 carries 8 significand bits, so the
/// recovered value is within 0.2% of a value that is at most 55.
pub fn assert_probe_row(
    fixture: &Fixture,
    got: &[bf16],
    step: u32,
    coordinate: u32,
    label: &str,
) {
    let steps = fixture.tape.config.steps(fixture.tape.cols) as usize;
    // The recurrence, not the window read: the two agreeing is the check.
    let states = fixture.tape.states_by_recurrence();
    let (hash_a, hash_b) = hash_params();
    let scale = codebook_scale();
    for (column, &value) in got.iter().enumerate() {
        let state = states[column * steps + step as usize];
        let expected = i32::from(state_codes(state, hash_a, hash_b)[coordinate as usize]);
        assert_eq!(
            (value.to_f32() / scale).round() as i32,
            expected,
            "{label} step={step} coord={coordinate} row={column}: state {state:#x} gave {value}, want {expected}"
        );
    }
}
