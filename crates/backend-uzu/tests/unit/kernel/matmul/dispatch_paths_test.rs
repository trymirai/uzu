#![cfg(metal_backend)]

use std::fmt::{Debug, Display};

use backend_uzu::{
    ArrayElement,
    backends::{
        common::{Backend, Context, kernel::ManualKernels, kernel::matmul::MatmulKernel},
        metal::{Metal, MetalContext},
    },
};
use proc_macros::__internal_uzu_test as uzu_test;
use half::{bf16, f16};
use num_traits::Float;

use crate::common::{
    assert::assert_eq_float,
    matmul::{Case, Variant, all_correctness_shapes, cpu_reference, deterministic_input, run_metal},
};

fn check_case<T: ArrayElement + Float + Debug + Display>(
    context: &MetalContext,
    kernel: &mut <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel,
    variant: Variant,
    case: Case,
    tolerance: f32,
) {
    let input = deterministic_input::<T>(case);
    let expected = cpu_reference::<T>(&input);
    let actual = run_metal::<T>(context, kernel, &input, variant);
    assert_eq_float(
        &expected,
        &actual,
        tolerance,
        &format!("{:?} dtype={} {:?}", variant, std::any::type_name::<T>(), case),
    );
}

fn check_all_shapes<T: ArrayElement + Float + Debug + Display>(
    variant: Variant,
    ab_scale: f32,
    accumulate: bool,
    tolerance: f32,
) {
    let context = MetalContext::new().expect("Metal context");
    if !variant.supported(&context) {
        eprintln!("Skipping {variant:?}: device does not support MXU");
        return;
    }
    let mut kernel = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("MatmulKernel");
    for shape in all_correctness_shapes() {
        let case = Case::new(shape).with_ab_scale(ab_scale).with_accumulate(accumulate);
        check_case::<T>(&context, &mut kernel, variant, case, tolerance);
    }
}

#[uzu_test]
fn matches_cpu_reference_bf16_gemm() {
    check_all_shapes::<bf16>(Variant::Gemm, 1.0, false, 1.0);
}

#[uzu_test]
fn matches_cpu_reference_bf16_gemm_mxu() {
    check_all_shapes::<bf16>(Variant::GemmMxu, 1.0, false, 1.0);
}

#[uzu_test]
fn matches_cpu_reference_f16_gemm() {
    check_all_shapes::<f16>(Variant::Gemm, 1.0, false, 0.5);
}

#[uzu_test]
fn matches_cpu_reference_f16_gemm_mxu() {
    check_all_shapes::<f16>(Variant::GemmMxu, 1.0, false, 0.5);
}

#[uzu_test]
fn matches_cpu_reference_f32_gemm() {
    check_all_shapes::<f32>(Variant::Gemm, 1.0, false, 0.05);
}

#[uzu_test]
fn ab_scale_bf16_gemm() {
    check_all_shapes::<bf16>(Variant::Gemm, 0.5, false, 1.0);
}

#[uzu_test]
fn ab_scale_bf16_gemm_mxu() {
    check_all_shapes::<bf16>(Variant::GemmMxu, 0.5, false, 1.0);
}

#[uzu_test]
fn accumulate_bf16_gemm_mxu() {
    check_all_shapes::<bf16>(Variant::GemmMxu, 1.0, true, 1.0);
}

#[uzu_test]
fn scale_and_accumulate_bf16_gemm_mxu() {
    check_all_shapes::<bf16>(Variant::GemmMxu, 0.5, true, 1.0);
}

fn check_b_transpose_false<T: ArrayElement + Float + Debug + Display>(
    variant: Variant,
    tolerance: f32,
) {
    let context = MetalContext::new().expect("Metal context");
    if !variant.supported(&context) {
        eprintln!("Skipping {variant:?}: device does not support MXU");
        return;
    }
    let mut kernel = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("MatmulKernel");
    for shape in all_correctness_shapes() {
        let case = Case {
            b_transpose: false,
            ..Case::new(shape)
        };
        check_case::<T>(&context, &mut kernel, variant, case, tolerance);
    }
}

#[uzu_test]
fn b_transpose_false_bf16_gemm() {
    check_b_transpose_false::<bf16>(Variant::Gemm, 1.0);
}

#[uzu_test]
fn b_transpose_false_bf16_gemm_mxu() {
    check_b_transpose_false::<bf16>(Variant::GemmMxu, 1.0);
}
