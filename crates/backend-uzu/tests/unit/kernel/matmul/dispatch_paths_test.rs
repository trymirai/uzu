#![cfg(metal_backend)]

use std::fmt::{Debug, Display};

use backend_uzu::{
    ArrayElement,
    backends::{
        common::{Backend, Context, kernel::ManualKernels, kernel::matmul::MatmulKernel},
        metal::{DeviceExt, GemmDispatchPath, Metal, MetalContext},
    },
};
use half::{bf16, f16};
use num_traits::Float;
use rstest::rstest;

use crate::common::{
    assert::assert_eq_float,
    matmul::{Case, all_correctness_shapes, cpu_reference, deterministic_input, run_metal},
};

fn gemm_paths_for_hw(context: &MetalContext) -> Vec<GemmDispatchPath> {
    let mut paths = vec![GemmDispatchPath::Simdgroup];
    if context.device.supports_mxu() {
        paths.push(GemmDispatchPath::Mxu);
    }
    paths
}

fn check_case<T: ArrayElement + Float + Debug + Display>(
    context: &MetalContext,
    kernel: &mut <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel,
    path: Option<GemmDispatchPath>,
    case: Case,
    tolerance: f32,
) {
    let input = deterministic_input::<T>(case);
    let expected = cpu_reference::<T>(&input);
    let actual = run_metal::<T>(context, kernel, &input, path);
    assert_eq_float(
        &expected,
        &actual,
        tolerance,
        &format!("{path:?} dtype={} {case:?}", std::any::type_name::<T>()),
    );
}

fn run_matrix<T: ArrayElement + Float + Debug + Display>(
    case_for_shape: impl Fn(crate::common::matmul::Shape) -> Case,
    tolerance: f32,
) {
    let context = MetalContext::new().expect("Metal context");
    let mut kernel = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("MatmulKernel");
    for path in gemm_paths_for_hw(&context) {
        for shape in all_correctness_shapes() {
            let case = case_for_shape(shape);
            check_case::<T>(&context, &mut kernel, Some(path), case, tolerance);
        }
    }
}

#[rstest]
#[case::base(1.0, false)]
#[case::ab_scale(0.5, false)]
#[case::accumulate(1.0, true)]
#[case::scale_and_accumulate(0.5, true)]
fn matches_cpu_reference_bf16(
    #[case] ab_scale: f32,
    #[case] accumulate: bool,
) {
    run_matrix::<bf16>(
        |shape| Case::new(shape).with_ab_scale(ab_scale).with_accumulate(accumulate),
        1.0,
    );
}

#[test]
fn matches_cpu_reference_f16() {
    run_matrix::<f16>(|shape| Case::new(shape), 0.5);
}

#[test]
fn matches_cpu_reference_f32() {
    run_matrix::<f32>(|shape| Case::new(shape), 0.01);
}

#[test]
fn b_transpose_false_bf16() {
    run_matrix::<bf16>(
        |shape| Case {
            b_transpose: false,
            ..Case::new(shape)
        },
        1.0,
    );
}
