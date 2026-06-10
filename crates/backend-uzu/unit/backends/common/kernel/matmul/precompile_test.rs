#![cfg(metal_backend)]

use half::bf16;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context,
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
            kernel::{
                Kernels,
                matmul::{MatmulKernel, MatmulTask},
            },
        },
        metal::{Metal, MetalContext},
    },
    common::{
        assert::assert_eq_float,
        matmul::{Case, Shape, cpu_reference, deterministic_input, run_metal},
    },
};

type Kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;

fn new_kernel(context: &MetalContext) -> Kernel {
    Kernel::new(context, bf16::data_type(), bf16::data_type(), bf16::data_type()).expect("MatmulKernel")
}

fn fp_task(
    n: u32,
    k: u32,
    d_transform: GemmDTransform,
) -> MatmulTask {
    MatmulTask {
        m: 0,
        n,
        k,
        b_transpose: true,
        b_offset: 0,
        b_leading_dimension: None,
        b_prologue: GemmBPrologueKind::FullPrecision,
        bits: None,
        group_size: None,
        d_transform,
    }
}

fn quant_task(
    b_prologue: GemmBPrologueKind,
    bits: u32,
    group_size: u32,
    n: u32,
    k: u32,
) -> MatmulTask {
    MatmulTask {
        m: 0,
        n,
        k,
        b_transpose: true,
        b_offset: 0,
        b_leading_dimension: None,
        b_prologue,
        bits: Some(bits),
        group_size: Some(group_size),
        d_transform: GemmDTransform::empty(),
    }
}

// After preheat, encode over GEMV (small m) and GEMM (large m) must still match
// the CPU reference.
#[test]
fn precompile_then_encode_matches_reference_bf16() {
    let context = MetalContext::new().expect("Metal context");
    let mut kernel = new_kernel(&context);

    let batch_sizes: Vec<u32> = (1..=72).collect();
    let projections = [(2048u32, 512u32), (512u32, 2048u32)];
    for &(n, k) in &projections {
        kernel.precompile(&context, &fp_task(n, k, GemmDTransform::empty()), &batch_sizes).expect("precompile");
    }

    for &m in &[1usize, 4, 8, 33, 64] {
        for &(n, k) in &projections {
            let case = Case::new(Shape::new(m, k as usize, n as usize));
            let input = deterministic_input::<bf16>(case);
            let expected = cpu_reference::<bf16>(&input);
            let actual = run_metal::<bf16>(&context, &mut kernel, &input, None);
            assert_eq_float(&expected, &actual, 1.0, &format!("preheated m={m} n={n} k={k}"));
        }
    }
}

// Every specialization the planner selects across the full batch domain must
// validate and compile.
#[test]
fn precompile_succeeds_across_batch_domain() {
    let context = MetalContext::new().expect("Metal context");
    let mut kernel = new_kernel(&context);

    let batch_sizes: Vec<u32> = (1..=72).collect();
    let tasks = [
        fp_task(2048, 512, GemmDTransform::BIAS),
        fp_task(2048, 512, GemmDTransform::RHT),
        quant_task(GemmBPrologueKind::ScaleBiasDequant, 4, 64, 2048, 512),
        quant_task(GemmBPrologueKind::ScaleZeroPointDequant, 8, 64, 2048, 512),
        quant_task(GemmBPrologueKind::ScaleSymmetricDequant, 4, 32, 2048, 512),
    ];
    for task in tasks {
        kernel.precompile(&context, &task, &batch_sizes).expect("precompile across batch domain");
    }
}
