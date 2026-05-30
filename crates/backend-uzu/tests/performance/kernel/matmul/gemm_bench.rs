#![cfg(metal_backend)]

use backend_uzu::{
    array::{ArrayContextExt, ArrayElement},
    backends::{
        common::{
            AllocationType, Backend, Context,
            kernel::{
                Kernels,
                matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        metal::{GemmDispatchPath, Metal, MetalContext},
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;

use crate::{
    common::{
        matmul::{bench_fp_gemm_shapes, iter_encode_loop},
        type_short_name,
    },
    uzu_bench,
};

#[uzu_bench]
fn bench_gemm(c: &mut Criterion) {
    let context = MetalContext::new().expect("Metal context");
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");

    let mut group = c.benchmark_group(format!("{}/Kernel/Matmul/GEMM", type_short_name::<Metal>()));

    for shape in bench_fp_gemm_shapes() {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let a = context
            .create_allocation(m * k * std::mem::size_of::<bf16>(), AllocationType::Global)
            .expect("a allocation");
        let b_array = context.create_array_uninitialized(&[n, k], bf16::data_type());
        let mut d = context
            .create_allocation(m * n * std::mem::size_of::<bf16>(), AllocationType::Global)
            .expect("d allocation");

        group.throughput(Throughput::Elements((2 * m * k * n) as u64));
        group.bench_function(BenchmarkId::new("BF16", shape.to_string()), |b| {
            iter_encode_loop::<Metal, _>(&context, b, |encoder| {
                kernel
                    .gemm
                    .encode_dispatch_path(
                        MatmulArguments {
                            a: &a,
                            a_offset: 0,
                            b: MatmulB::FullPrecision {
                                b: b_array.allocation(),
                            },
                            b_offset: 0,
                            b_leading_dimension: None,
                            b_transpose: true,
                            d: &mut d,
                            d_transform: MatmulDOps::none(),
                            m: m as u32,
                            n: n as u32,
                            k: k as u32,
                        },
                        GemmDispatchPath::Mxu,
                        encoder,
                    )
                    .expect("encode_dispatch_path failed");
            });
        });
    }
}
