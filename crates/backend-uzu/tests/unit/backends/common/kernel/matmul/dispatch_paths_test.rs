#![cfg(metal_backend)]

use std::fmt::{Debug, Display};

use half::bf16;
use num_traits::Float;
use proc_macros::uzu_test;
use rstest::rstest;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context,
            kernel::{Kernels, matmul::MatmulKernel},
        },
        metal::{GemmDispatchPath, Metal, MetalContext},
    },
    tests::{
        assert::assert_eq_float,
        matmul::{Case, all_correctness_shapes, cpu_reference, deterministic_input, run_metal},
    },
};

fn gemm_paths_for_hw(context: &MetalContext) -> Vec<GemmDispatchPath> {
    let mut paths = vec![GemmDispatchPath::Simdgroup];
    if context.supports_mxu() {
        paths.push(GemmDispatchPath::Mxu);
    }
    paths
}

fn check_case<T: ArrayElement + Float + Debug + Display>(
    context: &MetalContext,
    kernel: &mut <<Metal as Backend>::Kernels as Kernels>::MatmulKernel,
    path: Option<GemmDispatchPath>,
    case: Case,
    tolerance: f32,
) {
    let input = deterministic_input::<T>(case);
    let expected = cpu_reference::<T>(&input);
    let actual = run_metal::<T>(context, kernel, &input, path);
    assert_eq_float(&expected, &actual, tolerance, &format!("{path:?} dtype={} {case:?}", std::any::type_name::<T>()));
}

fn run_matrix<T: ArrayElement + Float + Debug + Display>(
    case_for_shape: impl Fn(crate::tests::matmul::Shape) -> Case,
    tolerance: f32,
) {
    let context = MetalContext::new().expect("Metal context");
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        T::data_type(),
        T::data_type(),
        T::data_type(),
    )
    .expect("MatmulKernel");
    for path in gemm_paths_for_hw(&context) {
        for shape in all_correctness_shapes() {
            let case = case_for_shape(shape);
            check_case::<T>(&context, &mut kernel, Some(path), case, tolerance);
        }
    }
}

#[rstest]
#[test_attr(uzu_test)]
#[case::base(1.0, false)]
#[case::ab_scale(0.5, false)]
#[case::accumulate(1.0, true)]
#[case::scale_and_accumulate(0.5, true)]
fn matches_cpu_reference_bf16(
    #[case] ab_scale: f32,
    #[case] accumulate: bool,
) {
    run_matrix::<bf16>(|shape| Case::new(shape).with_ab_scale(ab_scale).with_accumulate(accumulate), 1.0);
}

#[uzu_test]
fn matches_cpu_reference_f32() {
    run_matrix::<f32>(Case::new, 0.01);
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
    let context = MetalContext::new().expect("Metal context");
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");
    for path in gemm_paths_for_hw(&context) {
        for shape in rht_shapes() {
            let case = Case::new(shape).with_rht(true);
            check_case::<bf16>(&context, &mut kernel, Some(path), case, 1.0);
        }
    }
}

#[uzu_test]
fn bias_parity_bf16() {
    use crate::tests::matmul::Shape;
    let context = MetalContext::new().expect("Metal context");
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");
    let shapes = [Shape::new(64, 128, 64), Shape::new(128, 2048, 128), Shape::new(33, 128, 64)];
    for path in gemm_paths_for_hw(&context) {
        for shape in shapes {
            let case = Case::new(shape).with_bias(true);
            check_case::<bf16>(&context, &mut kernel, Some(path), case, 1.0);
        }
    }
}

#[uzu_test]
fn gemv_fp_partial_output_block_bf16() {
    use crate::tests::matmul::Shape;
    let context = MetalContext::new().expect("Metal context");
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");
    // Small m routes to GEMV; n that is not a multiple of the 32-row threadgroup
    // block exercises the tail-row clamp (must not read weight rows past B).
    let shapes = [Shape::new(1, 128, 33), Shape::new(2, 256, 65), Shape::new(4, 384, 100)];
    for shape in shapes {
        check_case::<bf16>(&context, &mut kernel, None, Case::new(shape), 1.0);
    }
}

#[uzu_test]
fn gemv_fp_output_transforms_bf16() {
    use crate::tests::matmul::Shape;
    let context = MetalContext::new().expect("Metal context");
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");
    // Small m routes to GEMV; n % 32 == 0 keeps RHT/accumulate in the GEMV path.
    let shape = Shape::new(2, 256, 64);
    let cases = [
        Case::new(shape).with_bias(true).with_rht(true),
        Case::new(shape).with_accumulate(true),
        Case::new(shape).with_accumulate(true).with_bias(true),
        Case::new(shape).with_ab_scale(2.0).with_rht(true),
    ];
    for case in cases {
        check_case::<bf16>(&context, &mut kernel, None, case, 1.0);
    }
}

#[uzu_test]
fn small_m_tiles_parity() {
    use crate::tests::matmul::Shape;
    let context = MetalContext::new().expect("Metal context");
    let mut bf16_kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");
    let mut f32_kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        f32::data_type(),
        f32::data_type(),
        f32::data_type(),
    )
    .expect("MatmulKernel");

    for path in gemm_paths_for_hw(&context) {
        for shape in [Shape::new(8, 256, 256), Shape::new(8, 512, 256)] {
            check_case::<bf16>(&context, &mut bf16_kernel, Some(path), Case::new(shape), 1.0);
            check_case::<f32>(&context, &mut f32_kernel, Some(path), Case::new(shape), 0.01);
        }
    }
}

/// Prints what the split-k heuristic actually resolves to across the a8w sweep grid.
///
/// Pure arithmetic, no GPU: a sweep over `SPLIT_K_TARGET_TILES_INT8_ACTIVATIONS` is only
/// interpretable if we know which (target, m) pairs genuinely differ. Targets that resolve
/// to the same split-k must produce the same kernel, so any measured gap between them is
/// measurement error rather than a tuning result.
#[uzu_test]
fn a8w_split_k_resolution_table() {
    use crate::backends::metal::{select_mxu_quant_tiling, select_split_k};

    const GROUP_SIZE: u32 = 64;
    for (label, k, n) in [("k2048_n3072", 2048u32, 3072u32), ("k3584_n1024", 3584, 1024)] {
        println!("\n{label}");
        for m in [16u32, 32, 64, 128, 256, 512, 1024, 2048] {
            let tiling = select_mxu_quant_tiling(m, n, GROUP_SIZE, true);
            let base_tiles = n.div_ceil(tiling.block_n()) * m.div_ceil(tiling.block_m());
            let splits: Vec<u32> = [128u32, 256, 512, 1024]
                .into_iter()
                .map(|target| select_split_k(m, n, k, tiling, true, GROUP_SIZE, false, false, target))
                .collect();
            println!("  m{m:<5} {tiling:?} base_tiles={base_tiles:<5} split_k(128/256/512/1024)={:?}", splits);
        }
    }
}
