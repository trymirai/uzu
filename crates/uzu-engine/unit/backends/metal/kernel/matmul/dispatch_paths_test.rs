use std::fmt::{Debug, Display};

use futures::executor::block_on;
use half::bf16;
use num_traits::Float;
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Context,
            kernel::{Kernels, matmul::MatmulKernel},
        },
        metal::{Metal, context::MetalContext, kernel::matmul::gemm::GemmEngine},
    },
    data_type::DataTypeRepr,
    tests::{
        assert::assert_eq_float,
        matmul::{
            Case, Shape, all_correctness_shapes, cpu_reference, deterministic_input,
            harness::{Input, run},
        },
    },
};

fn run_metal<T: DataTypeRepr + Float>(
    context: &MetalContext,
    kernel: &mut <<Metal as Backend>::Kernels as Kernels>::MatmulKernel,
    input: &Input<T>,
    dispatch: Option<GemmEngine>,
) -> Vec<T> {
    run::<Metal, T>(context, kernel, input, |kernel, args, command_buffer| {
        if let Some(engine) = dispatch {
            kernel.encode_with_gemm_engine(args, engine, command_buffer).expect("forced GEMM engine encode failed");
        } else {
            kernel.encode(args, command_buffer).expect("matmul encode failed");
        }
    })
}

fn gemm_engines_for_hw(context: &MetalContext) -> Vec<GemmEngine> {
    let mut engines = vec![GemmEngine::Simdgroup];
    if context.supports_mxu {
        engines.push(GemmEngine::Mxu);
    }
    engines
}

fn check_case<T: DataTypeRepr + Float + Debug + Display>(
    context: &MetalContext,
    kernel: &mut <<Metal as Backend>::Kernels as Kernels>::MatmulKernel,
    dispatch: Option<GemmEngine>,
    case: Case,
    tolerance: f32,
) {
    let input = deterministic_input::<T>(case);
    let expected = cpu_reference::<T>(&input);
    let actual = run_metal::<T>(context, kernel, &input, dispatch);
    assert_eq_float(
        &expected,
        &actual,
        tolerance,
        &format!("{dispatch:?} dtype={} {case:?}", std::any::type_name::<T>()),
    );
}

fn run_matrix<T: DataTypeRepr + Float + Debug + Display>(
    case_for_shape: impl Fn(crate::tests::matmul::Shape) -> Case,
    tolerance: f32,
) {
    let context = block_on(MetalContext::new()).expect("Metal context");
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        T::data_type(),
        T::data_type(),
        T::data_type(),
    )
    .expect("MatmulKernel");
    for engine in gemm_engines_for_hw(&context) {
        for shape in all_correctness_shapes() {
            let case = case_for_shape(shape);
            check_case::<T>(&context, &mut kernel, Some(engine), case, tolerance);
        }
    }
}

#[rstest]
#[test_attr(uzu_test)]
#[case::base(1.0)]
#[case::ab_scale(0.5)]
fn matches_cpu_reference_bf16(#[case] ab_scale: f32) {
    run_matrix::<bf16>(|shape| Case::new(shape).with_ab_scale(ab_scale), 1.0);
}

#[uzu_test]
fn b_transpose_false_bf16() {
    run_matrix::<bf16>(
        |shape| Case {
            b_transpose: false,
            ..Case::new(shape)
        },
        1.0,
    );
}

fn rht_shapes() -> impl Iterator<Item = crate::tests::matmul::Shape> {
    use crate::tests::matmul::Shape;
    [Shape::new(8, 128, 64), Shape::new(64, 128, 128), Shape::new(128, 2048, 256), Shape::new(33, 128, 64)].into_iter()
}

#[uzu_test]
fn rht_parity_bf16() {
    let context = block_on(MetalContext::new()).expect("Metal context");
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");
    for engine in gemm_engines_for_hw(&context) {
        for shape in rht_shapes() {
            let case = Case::new(shape).with_rht(true);
            check_case::<bf16>(&context, &mut kernel, Some(engine), case, 1.0);
        }

        let split_k_case = Case::new(Shape::new(4, 1024, 32)).with_rht(true).with_bias(true);
        check_case::<bf16>(&context, &mut kernel, Some(engine), split_k_case, 0.1);

        let non_split_case = Case::new(Shape::new(64, 128, 128)).with_rht(true).with_bias(true);
        check_case::<bf16>(&context, &mut kernel, Some(engine), non_split_case, 0.1);
    }
}

#[uzu_test]
fn bias_parity_bf16() {
    use crate::tests::matmul::Shape;
    let context = block_on(MetalContext::new()).expect("Metal context");
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");
    let shapes = [Shape::new(64, 128, 64), Shape::new(128, 2048, 128), Shape::new(33, 128, 64)];
    for engine in gemm_engines_for_hw(&context) {
        for shape in shapes {
            let case = Case::new(shape).with_bias(true);
            check_case::<bf16>(&context, &mut kernel, Some(engine), case, 1.0);
        }
    }
}

#[uzu_test]
fn gemv_fp_partial_output_block_bf16() {
    let context = block_on(MetalContext::new()).expect("Metal context");
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");
    let shapes = [Shape::new(1, 128, 33), Shape::new(2, 256, 65), Shape::new(4, 384, 100)];
    for shape in shapes {
        check_case::<bf16>(&context, &mut kernel, None, Case::new(shape), 1.0);
    }
}

#[uzu_test]
fn gemv_fp_output_transforms_bf16() {
    use crate::tests::matmul::Shape;
    let context = block_on(MetalContext::new()).expect("Metal context");
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");
    // Small m routes to GEMV; n % 32 == 0 keeps RHT in the GEMV path.
    let shape = Shape::new(2, 256, 64);
    let cases = [Case::new(shape).with_bias(true).with_rht(true), Case::new(shape).with_ab_scale(2.0).with_rht(true)];
    for case in cases {
        check_case::<bf16>(&context, &mut kernel, None, case, 1.0);
    }

    // M5 selects the ordinary 8-row, four-slice FP GEMV tile; RHT and bias are deferred.
    let split_k_rht = Case::new(Shape::new(1, 512, 1024)).with_rht(true).with_bias(true);
    check_case::<bf16>(&context, &mut kernel, None, split_k_rht, 1.0);
}

#[uzu_test]
fn small_m_square_tile_parity() {
    use crate::tests::matmul::Shape;
    let context = block_on(MetalContext::new()).expect("Metal context");
    let mut bf16_kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");
    for shape in [Shape::new(8, 256, 256), Shape::new(8, 512, 256)] {
        for engine in gemm_engines_for_hw(&context) {
            check_case::<bf16>(&context, &mut bf16_kernel, Some(engine), Case::new(shape), 1.0);
        }
    }
}
