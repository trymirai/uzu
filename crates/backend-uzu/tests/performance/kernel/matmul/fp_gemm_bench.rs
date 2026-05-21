//! FP (non-MXU) GEMM throughput bench. Drives `MatmulDispatchPath::Gemm`.
//!
//! Shapes and bench-ID format intentionally mirror the pre-unification
//! `bench_gemm` in commit `26d23f9` (`tests/unit/kernel/matmul/gemm_test.rs`)
//! so a worktree at that commit and this branch can be compared 1:1.

#![cfg(metal_backend)]

use std::collections::HashSet;

use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{
            AllocationType, Backend, Context, Encoder,
            kernel::{
                ManualKernels,
                matmul::{MatmulArguments, MatmulB, MatmulKernel},
            },
        },
        metal::{Metal, MatmulDispatchPath, MetalContext},
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;

use crate::{common::type_short_name, uzu_bench};

const BENCHMARK_SHAPES: &[(usize, usize, usize)] = &[
    (128, 2048, 8192),
    (128, 4096, 14336),
    (256, 4096, 4096),
    (512, 8192, 2048),
];

#[uzu_bench]
fn bench_gemm(c: &mut Criterion) {
    let context = MetalContext::new().expect("Metal context");

    let mut kernel =
        <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, bf16::data_type())
            .expect("MatmulKernel");

    let mut group =
        c.benchmark_group(format!("{}/Kernel/Matmul/GEMM", type_short_name::<Metal>()));

    for &(m, k, n) in BENCHMARK_SHAPES {
        let a = context
            .create_allocation(m * k * std::mem::size_of::<bf16>(), AllocationType::Global)
            .expect("a allocation");
        let b_array = context.create_array_uninitialized(&[n, k], bf16::data_type());
        let mut d = context
            .create_allocation(m * n * std::mem::size_of::<bf16>(), AllocationType::Global)
            .expect("d allocation");

        group.throughput(Throughput::Elements((2 * m * k * n) as u64));
        group.bench_function(
            BenchmarkId::new("BF16", format!("M[{m}]K[{k}]N[{n}]")),
            |bencher| {
                bencher.iter_custom(|n_iters| {
                    let mut encoder = Encoder::<Metal>::new(&context).unwrap();
                    for _ in 0..n_iters {
                        kernel
                            .encode_with_path(
                                MatmulArguments {
                                    a: &a,
                                    a_offset: 0,
                                    a_prologue: HashSet::new(),
                                    b: MatmulB::FullPrecision {
                                        b: b_array.allocation(),
                                    },
                                    b_offset: 0,
                                    b_leading_dimension: None,
                                    b_transpose: true,
                                    d: &mut d,
                                    d_transform: HashSet::new(),
                                    m: m as u32,
                                    n: n as u32,
                                    k: k as u32,
                                },
                                &mut encoder,
                                MatmulDispatchPath::Gemm,
                            )
                            .expect("encode_with_path failed");
                    }
                    encoder
                        .end_encoding()
                        .submit()
                        .wait_until_completed()
                        .unwrap()
                        .gpu_execution_time()
                })
            },
        );
    }
}
