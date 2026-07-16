#![cfg(metal_backend)]

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend,
            kernel::{
                Kernels,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        metal::{GemmDispatchPath, Metal},
    },
    tests::{
        helpers::alloc_allocation,
        matmul::{bench_fp_gemm_shapes, iter_encode_loop},
        util::type_short_name,
    },
};

#[uzu_bench]
fn bench_gemm(c: &mut Criterion) {
    let context = crate::tests::util::shared_metal_context();
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");

    let paths: &[(&str, GemmDispatchPath)] = if context.supports_mxu() {
        &[("GEMM", GemmDispatchPath::Simdgroup), ("GEMM_MXU", GemmDispatchPath::Mxu)]
    } else {
        &[("GEMM", GemmDispatchPath::Simdgroup)]
    };

    for &(group_label, path) in paths {
        let mut group = c.benchmark_group(format!("{}/Kernel/Matmul/{}", type_short_name::<Metal>(), group_label));

        for shape in bench_fp_gemm_shapes() {
            let (m, k, n) = (shape.m, shape.k, shape.n);
            let a = alloc_allocation::<Metal, bf16>(&context, m * k);
            let b_weights = alloc_allocation::<Metal, bf16>(&context, n * k);
            let mut d = alloc_allocation::<Metal, bf16>(&context, m * n);

            group.throughput(Throughput::Elements((2 * m * k * n) as u64));
            group.bench_function(BenchmarkId::new("BF16", shape.to_string()), |b| {
                iter_encode_loop::<Metal, _>(&context, b, |encoder| {
                    kernel
                        .gemm
                        .encode_dispatch_path(
                            MatmulArguments {
                                a: MatmulA::FullPrecision {
                                    values: &a,
                                    offset: 0,
                                },
                                b: MatmulB::FullPrecision {
                                    b: &b_weights,
                                },
                                b_leading_dimension: None,
                                b_transpose: true,
                                d: &mut d,
                                d_transform: MatmulDOps::none(),
                                m: m as u32,
                                n: n as u32,
                                k: k as u32,
                            },
                            path,
                            encoder,
                        )
                        .expect("encode_dispatch_path failed");
                });
            });
        }
    }
}
