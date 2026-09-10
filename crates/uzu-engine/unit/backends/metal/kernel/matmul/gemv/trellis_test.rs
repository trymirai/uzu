//! The trellis GEMV on the production `MatmulKernel` dispatch.
//!
//! Three checks, and they fail for different reasons: a numeric one against the
//! shared f64 oracle, swept over every compiled batch width, a ragged `N`, a `K`
//! that is and is not a power of two, and every window width the format allows;
//! the shared bit-exact probe, which pins the kernel's 128-bit run read and the
//! tape packing against the trellis recurrence with no tolerance at all; and
//! routing, which is visible from outside as the activation format -- the GEMV
//! takes bf16 activations, the trellis GEMM takes int8 ones.
//!
//! `L` and `KV` are runtime scalars, so one config cannot exercise them:
//! [`trellis_gemv_spans_every_window`] sweeps the narrowest window this kernel
//! sees up to `KV == L == 32`, where the lane's four-state run spans exactly the
//! 128 bits `TrellisSlice` holds.
//!
//! TOLERANCE. `D` is bf16 (the trellis GEMV family is pinned to bf16 in and
//! out), and bf16 carries 8 significand bits, so simply STORING a correct result
//! costs up to `2^-8 = 3.9e-3` relative on its own. The bar below is therefore
//! set against the oracle ROUNDED TO bf16, where the measured error is exactly
//! ZERO at every M, K and window width -- the kernel reproduces the oracle's
//! bf16 bit for bit, which is a strictly stronger statement than any tolerance
//! against the f64 oracle. The oracle reads the activations as the bf16 values
//! the GPU read, so activation rounding is not in the number.

use half::bf16;
use uzu_engine_macros::uzu_test;

use super::*;
use crate::{
    backends::{
        common::{
            Encoder,
            kernel::matmul::{
                ActivationFormat, MatmulA, MatmulArguments, MatmulDOps, MatmulKernel, MatmulShape,
                trellis_format::{TRELLIS_BLOCK_K, TrellisConfig},
            },
        },
        metal::{Metal, context::MetalContext, kernel::matmul::MatmulMetalKernel},
    },
    tests::{
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
        matmul::trellis_fixture::{
            CONFIG, Fixture, assert_probe_row, max_relative_error, oracle, probe_points, row_scales,
        },
        util::shared_metal_context,
    },
};

/// Ragged against the 16-row output tile.
const N: u32 = 130;

/// `d = a * weights^T` through the production dispatch, with bf16 activations.
fn run(
    context: &MetalContext,
    fixture: &Fixture,
    a: &[bf16],
    m: u32,
    n: u32,
    k: u32,
) -> Vec<bf16> {
    let a_buffer = alloc_allocation_with_data::<Metal, bf16>(context, a);
    let mut d = alloc_allocation::<Metal, bf16>(context, (m * n) as usize);

    let arguments = MatmulArguments {
        a: MatmulA::FullPrecision {
            values: &a_buffer,
            offset: 0,
        },
        b: fixture.weights(),
        b_leading_dimension: None,
        b_transpose: true,
        d: &mut d,
        d_transform: MatmulDOps::none(),
        gather_indices: None,
        m,
        n,
        k,
    };

    let mut kernel =
        MatmulMetalKernel::new(context, DataType::BF16, DataType::BF16, DataType::BF16).expect("MatmulMetalKernel");
    let mut encoder = Encoder::<Metal>::new(context).expect("encoder");
    kernel.encode(arguments, &mut encoder).expect("trellis GEMV encode failed");
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec::<Metal, bf16>(&d)
}

/// Deterministic bf16 activations.
fn activations(
    m: u32,
    k: u32,
) -> Vec<bf16> {
    (0..m * k).map(|i| bf16::from_f32(((i % 23) as f32 - 11.0) / 32.0)).collect()
}

/// The kernel against the oracle's own bf16 rounding; see the TOLERANCE note.
fn assert_matches_oracle(
    fixture: &Fixture,
    got: &[bf16],
    a: &[bf16],
    m: u32,
    n: u32,
    k: u32,
    label: &str,
) {
    let expected = oracle(fixture, m, n, k, |row, index| f64::from(a[row * k as usize + index].to_f32()));
    let rounded: Vec<f32> = expected.iter().map(|&value| bf16::from_f32(value).to_f32()).collect();
    let error = max_relative_error(&got[..(m * n) as usize], &rounded);
    assert!(error < 1e-3, "{label}: max relative error {error:e} against the bf16-rounded oracle");
}

#[uzu_test]
fn trellis_gemv_matches_oracle() {
    let context = shared_metal_context();
    // 512 is a power of two; 1152 is nine 128-column blocks and is NOT a
    // multiple of the 256 the INT8 GEMV calls aligned, which is what the
    // trellis K block replaces.
    for k in [512u32, 1152] {
        let fixture = Fixture::new(&context, CONFIG, N, k, row_scales(N));
        for m in 1..=8u32 {
            let a = activations(m, k);
            let got = run(&context, &fixture, &a, m, N, k);
            assert_matches_oracle(&fixture, &got, &a, m, N, k, &format!("K={k} M={m}"));
        }
    }
}

/// `L` and `KV` are runtime scalars; this is the sweep that exercises them,
/// including `KV == L == 32`, where a lane's four-state run spans exactly the
/// 128 bits the slice holds.
#[uzu_test]
fn trellis_gemv_spans_every_window() {
    let context = shared_metal_context();
    let k = 512u32;
    let n = 33u32;
    for config in [
        TrellisConfig::new(16, 2),
        TrellisConfig::new(24, 3),
        TrellisConfig::new(28, 3),
        TrellisConfig::new(32, 3),
        TrellisConfig::new(32, 8),
    ] {
        let fixture = Fixture::new(&context, config, n, k, row_scales(n));
        for m in [1u32, 4, 8] {
            let a = activations(m, k);
            let got = run(&context, &fixture, &a, m, n, k);
            assert_matches_oracle(&fixture, &got, &a, m, n, k, &format!("{config:?} M={m}"));
        }
    }
}

#[uzu_test]
fn trellis_gemv_states_are_bit_exact() {
    let context = shared_metal_context();
    // Long enough that the probe reaches steps whose funnel shift is misaligned
    // in every one of the 32 possible ways.
    let k = 1024u32;
    let fixture = Fixture::new(&context, CONFIG, N, k, vec![bf16::ONE; N as usize]);

    for &(step, coordinate) in probe_points(CONFIG.steps(k)).iter() {
        let mut a = vec![bf16::ZERO; k as usize];
        a[(step * 4 + coordinate) as usize] = bf16::ONE;
        let got = run(&context, &fixture, &a, 1, N, k);
        assert_probe_row(&fixture, &got[..N as usize], step, coordinate, "gemv");
    }
}

#[uzu_test]
fn trellis_gemv_routes_by_batch() {
    let context = shared_metal_context();
    let kernel =
        MatmulMetalKernel::new(&context, DataType::BF16, DataType::BF16, DataType::BF16).expect("MatmulMetalKernel");
    let select = |shape: &MatmulShape| {
        GemvSpecialization::select_shape(
            shape,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            context.gpu_core_count,
            context.apple_gpu_family,
        )
    };
    for m in 1..=8u32 {
        let shape = trellis_shape(m, 16480, 5120);
        let specialization = select(&shape).unwrap_or_else(|| panic!("M={m} must reach the trellis GEMV"));
        // The batch split: eight output rows at M <= 2, sixteen above it, over
        // eight simdgroups of 32 reduction lanes either way. See
        // `GemvSpecialization::select_shape`.
        assert_eq!(
            specialization.output_row_tile(),
            if m <= 2 {
                8
            } else {
                16
            },
            "M={m}"
        );
        // Thirty-two reduction lanes up to M = 4, eight above it.
        assert_eq!(
            specialization.reduction_lanes(),
            if m <= 4 {
                32
            } else {
                8
            },
            "M={m}"
        );
        assert_eq!(
            kernel.select_activation_format(&shape, &context),
            ActivationFormat::Bf16,
            "M={m} must keep bf16 activations"
        );
    }
    for m in [9u32, 16, 64] {
        let shape = trellis_shape(m, 16480, 5120);
        assert!(select(&shape).is_none(), "M={m} must fall to the GEMM");
        if context.supports_mxu {
            assert_eq!(
                kernel.select_activation_format(&shape, &context),
                ActivationFormat::Int8,
                "M={m} must reach the int8 trellis GEMM"
            );
        }
    }
    // K that is not a whole number of trellis K blocks has no GEMV variant.
    assert!(select(&trellis_shape(4, 16480, 5120 + 64)).is_none());
    // K that is a whole number of 128s but not of the 32-lane block's 512 falls
    // back to the eight-lane tile instead of off the GEMV -- at EVERY batch
    // width, because that tile walks the batch through the grid rather than
    // compiling an input row tile of its own.
    for m in 1..=8u32 {
        let specialization = select(&trellis_shape(m, 16480, 5120 + 128)).expect("eight-lane fallback");
        assert_eq!(specialization.output_row_tile(), 16, "M={m} K=5248");
        assert_eq!(specialization.reduction_lanes(), 8, "M={m} K=5248");
    }
}

fn trellis_shape(
    m: u32,
    n: u32,
    k: u32,
) -> MatmulShape {
    MatmulShape {
        m,
        n,
        k,
        b_transpose: true,
        b_leading_dimension: None,
        b_prologue: GemmBPrologueKind::Trellis,
        b_bits: Some(8),
        b_group_size: Some(TRELLIS_BLOCK_K),
        signed_codes: true,
        a_full_precision: true,
        gathered: false,
        d_transform: GemmDTransform::empty(),
    }
}
