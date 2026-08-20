#![cfg(backend = "metal")]

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
        metal::{GemmEngine, Metal},
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

    let engines: &[(&str, GemmEngine)] = if context.supports_mxu() {
        &[("GEMM", GemmEngine::Simdgroup), ("GEMM_MXU", GemmEngine::Mxu)]
    } else {
        &[("GEMM", GemmEngine::Simdgroup)]
    };

    for &(group_label, engine) in engines {
        let mut group = c.benchmark_group(format!("{}/Kernel/Matmul/{}", type_short_name::<Metal>(), group_label));

        for shape in bench_fp_gemm_shapes() {
            let (m, k, n) = (shape.m, shape.k, shape.n);
            let a = alloc_allocation::<Metal, bf16>(&context, m as usize * k as usize);
            let b_weights = alloc_allocation::<Metal, bf16>(&context, n as usize * k as usize);
            let mut d = alloc_allocation::<Metal, bf16>(&context, m as usize * n as usize);
            group.throughput(Throughput::Elements(2 * u64::from(m) * u64::from(k) * u64::from(n)));
            group.bench_function(BenchmarkId::new("BF16", shape.to_string()), |b| {
                iter_encode_loop::<Metal, _>(&context, b, |encoder| {
                    kernel
                        .gemm
                        .encode_with_engine(
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
                                gather_indices: None,
                                expert_routes: None,
                                m,
                                n,
                                k,
                            },
                            engine,
                            encoder,
                        )
                        .expect("encode_plan failed");
                });
            });
        }
    }
}
