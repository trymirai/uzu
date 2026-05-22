#![cfg(metal_backend)]

use std::{
    collections::HashSet,
    error::Error as StdError,
    fmt::{Debug, Display},
};

use backend_uzu::{
    ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder,
            gpu_types::{QuantizationMethod, gemm::GemmDTransform},
            kernel::{
                ManualKernels,
                matmul::{MatmulDOp, MatmulError, MatmulKernel},
            },
        },
        metal::{MatmulDispatchPath, Metal, MetalContext},
    },
};
use half::bf16;
use num_traits::Float;
use proc_macros::__internal_uzu_test as uzu_test;
use rstest::rstest;

use crate::common::{
    helpers::allocation_to_vec,
    matmul::{QuantBuffers, QuantInput, quant_arguments, run_quant_cpu, run_quant_metal},
};

fn check_tolerance(
    expected: f32,
    actual: f32,
    rel_tol: f64,
    abs_tol: f64,
) -> bool {
    let diff = (expected - actual).abs() as f64;
    let tol = abs_tol.max(expected.abs() as f64 * rel_tol);
    diff <= tol
}

fn assert_parity<T: ArrayElement + Float + Debug + Display>(
    label: &str,
    reference: &[T],
    actual: &[T],
    rel_tol: f64,
    abs_tol: f64,
) {
    assert_eq!(reference.len(), actual.len(), "{label}: length mismatch");
    let mut errors = 0usize;
    for (i, (&exp, &got)) in reference.iter().zip(actual.iter()).enumerate() {
        let exp_f32 = exp.to_f32().unwrap();
        let got_f32 = got.to_f32().unwrap();
        if !check_tolerance(exp_f32, got_f32, rel_tol, abs_tol) {
            if errors < 5 {
                eprintln!("  {label} idx={i} expected={exp_f32} got={got_f32} diff={}", (exp_f32 - got_f32).abs());
            }
            errors += 1;
        }
    }
    assert_eq!(errors, 0, "{label}: {errors} mismatches");
}

fn run_parity<T: ArrayElement + Float + Debug + Display>(
    m: usize,
    k: usize,
    n: usize,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
    rel_tol: f64,
    abs_tol: f64,
) {
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 0);
    let unified = run_quant_metal::<T>(&context, &input, MatmulDispatchPath::QuantGemm);
    let reference = run_quant_cpu::<T>(&input);

    assert_parity::<T>(
        &format!(
            "m={m} k={k} n={n} gs={group_size} bits={bits} method={quant_method:?} dtype={}",
            std::any::type_name::<T>()
        ),
        &reference,
        &unified,
        rel_tol,
        abs_tol,
    );
}

// --- bf16: parity vs CPU backend across (group_size, bits, method, shape) ---
// Test inputs are deliberately small-magnitude (see QuantInput::new) so that
// the bf16 multiply-add drift accumulates to ~0.3 absolute. This lets us hold
// a tight (rel=0.05, abs=0.4) tolerance — a bug that scales results by >5%
// or shifts them by >0.4 won't slip through.
#[rstest]
//                            (   m,    k,   n,   gs, bits, method)
#[case::gs32_4bit_mlx_prefill ( 64, 256,  64,  32, 4, QuantizationMethod::ScaleBias)]
#[case::gs64_4bit_mlx_prefill ( 64, 256,  64,  64, 4, QuantizationMethod::ScaleBias)]
#[case::gs128_4bit_mlx_prefill( 64, 256,  64, 128, 4, QuantizationMethod::ScaleBias)]
#[case::gs32_8bit_mlx_prefill ( 64, 256,  64,  32, 8, QuantizationMethod::ScaleBias)]
#[case::gs64_8bit_zp_prefill  ( 64, 256,  64,  64, 8, QuantizationMethod::ScaleZeroPoint)]
#[case::gs128_8bit_zp_prefill ( 64, 256,  64, 128, 8, QuantizationMethod::ScaleZeroPoint)]
#[case::gs32_4bit_mlx_decode  (  8, 256,  64,  32, 4, QuantizationMethod::ScaleBias)]
#[case::gs64_4bit_zp_decode   (  8, 256,  64,  64, 4, QuantizationMethod::ScaleZeroPoint)]
#[case::gs32_unaligned_n      ( 64, 256,  96,  32, 4, QuantizationMethod::ScaleBias)]
fn parity_bf16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    run_parity::<bf16>(m, k, n, gs, bits, method, 0.05, 0.4);
}

// --- f16: exercises (32,32,32) tile only ---
#[rstest]
#[case::gs64_4bit_mlx (32, 256, 64,  64, 4, QuantizationMethod::ScaleBias)]
#[case::gs128_8bit_zp (32, 256, 64, 128, 8, QuantizationMethod::ScaleZeroPoint)]
fn parity_f16(
    #[case] m: usize,
    #[case] k: usize,
    #[case] n: usize,
    #[case] gs: u32,
    #[case] bits: u32,
    #[case] method: QuantizationMethod,
) {
    run_parity::<half::f16>(m, k, n, gs, bits, method, 0.01, 0.05);
}

// Bias post-pass for quant_gemm: a MatmulDOp::Bias is applied as a separate
// bias-add kernel after the matmul core. Reference adds the same broadcast bias.
#[uzu_test]
fn parity_bf16_gs32_4bit_mlx_with_bias() {
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<bf16>::new(64, 256, 64, 32, 4, QuantizationMethod::ScaleBias, 0);

    let bias_f32: Vec<f32> = (0..input.n as usize).map(|j| 0.5 + 0.1 * (j % 5) as f32).collect();
    let bias_t: Vec<bf16> = bias_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let bias_pp_buf =
        crate::common::helpers::alloc_allocation_with_data::<Metal, bf16>(&context, &bias_t);
    let mut matmul = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, bf16::data_type())
        .expect("MatmulMetalKernel");

    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
    let mut args = quant_arguments(&mut buffers, &input);
    args.d_transform = HashSet::from([MatmulDOp::Bias {
        bias: &bias_pp_buf,
    }]);
    matmul
        .encode_with_path(args, &mut encoder, MatmulDispatchPath::QuantGemm)
        .expect("encode quant with bias");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let actual = allocation_to_vec::<Metal, bf16>(&buffers.y);

    let mut reference = run_quant_cpu::<bf16>(&input);
    for i in 0..(input.m as usize) {
        for j in 0..(input.n as usize) {
            let idx = i * input.n as usize + j;
            reference[idx] = bf16::from_f32(reference[idx].to_f32() + bias_f32[j]);
        }
    }
    assert_parity::<bf16>("with_bias", &reference, &actual, 0.05, 0.4);
}

// Regression: quant dispatch rejects nonzero b_offset rather than silently dropping it.
#[uzu_test]
fn quant_gemm_nonzero_b_offset_returns_unsupported_layout() {
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<bf16>::new(64, 256, 64, 32, 4, QuantizationMethod::ScaleBias, 0);
    let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let mut matmul = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, bf16::data_type())
        .expect("MatmulMetalKernel");

    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
    let mut args = quant_arguments(&mut buffers, &input);
    args.b_offset = 16;
    let result = matmul.encode_with_path(args, &mut encoder, MatmulDispatchPath::QuantGemm);

    let err = result.expect_err("expected error");
    let matmul: &MatmulError<Metal> = (&err as &dyn StdError)
        .source()
        .and_then(|s| s.downcast_ref::<MatmulError<Metal>>())
        .expect("expected MatmulError source");
    assert!(
        matches!(matmul, MatmulError::UnsupportedLayout { path: "QuantGemm" }),
        "got {matmul:?}"
    );
}

// Regression: encode returns Err on rejected D-transform instead of panicking.
#[uzu_test]
fn quant_gemm_accumulate_returns_unsupported_dop() {
    let context = MetalContext::new().expect("Metal context");
    let input = QuantInput::<bf16>::new(64, 256, 64, 32, 4, QuantizationMethod::ScaleBias, 0);
    let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
    let mut matmul = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, bf16::data_type())
        .expect("MatmulMetalKernel");

    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
    let mut args = quant_arguments(&mut buffers, &input);
    args.d_transform = HashSet::from([MatmulDOp::Accumulate]);
    let result = matmul.encode_with_path(args, &mut encoder, MatmulDispatchPath::QuantGemm);

    let err = result.expect_err("expected error");
    let matmul: &MatmulError<Metal> = (&err as &dyn StdError)
        .source()
        .and_then(|s| s.downcast_ref::<MatmulError<Metal>>())
        .expect("expected MatmulError source");
    assert!(
        matches!(
            matmul,
            MatmulError::UnsupportedDOp {
                bit: GemmDTransform::ACCUMULATE,
                ..
            }
        ),
        "got {matmul:?}"
    );
}
