#![cfg(metal_backend)]

//! Full-precision (bf16) GEMV GPU-time bench (mirrors `quant_gemv_bench` for the
//! `MatmulB::FullPrecision` path), reporting `gpu_execution_time`.

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
        metal::Metal,
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;

use crate::{
    common::{
        matmul::{bench_quant_gemv_shapes, iter_encode_loop},
        type_short_name,
    },
    uzu_bench,
};

#[uzu_bench]
fn bench_fp_gemv(c: &mut Criterion) {
    let context = crate::common::shared_metal_context();
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
    )
    .expect("MatmulKernel");

    let mut group = c.benchmark_group(format!("{}/Kernel/GemvFP/BF16", type_short_name::<Metal>()));

    for shape in bench_quant_gemv_shapes(4) {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let a = context
            .create_allocation(m * k * std::mem::size_of::<bf16>(), AllocationType::Global)
            .expect("a allocation");
        let b_array = context.create_array_uninitialized(&[n, k], bf16::data_type());
        let mut d = context
            .create_allocation(m * n * std::mem::size_of::<bf16>(), AllocationType::Global)
            .expect("d allocation");

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(shape.to_string()), |b| {
            iter_encode_loop::<Metal, _>(&context, b, |encoder| {
                kernel
                    .encode(
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
                        encoder,
                    )
                    .expect("fp gemv encode");
            });
        });
    }
}
