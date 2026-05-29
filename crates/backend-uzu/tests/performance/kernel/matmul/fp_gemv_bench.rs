#![cfg(metal_backend)]

use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{
            AllocationType, Backend, Context,
            kernel::{
                ManualKernels,
                matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        metal::{Metal, MetalContext},
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use num_traits::Float;

use crate::{
    common::{matmul::iter_encode_loop, type_short_name},
    uzu_bench,
};

const ALL_SHAPES: &[(&str, usize, usize)] = &[
    ("qkv_0.6b", 1024, 4096),
    ("o_0.6b", 2048, 1024),
    ("gateup_0.6b", 1024, 6144),
    ("down_0.6b", 3072, 1024),
    ("qkv_1.7b", 2048, 4096),
    ("o_1.7b", 2048, 2048),
    ("gateup_1.7b", 2048, 12288),
    ("down_1.7b", 6144, 2048),
    ("qkv_4b", 2560, 6144),
    ("o_4b", 4096, 2560),
    ("gateup_4b", 2560, 19456),
    ("down_4b", 9728, 2560),
    ("qkv_8b", 4096, 6144),
    ("o_8b", 4096, 4096),
    ("gateup_8b", 4096, 24576),
    ("down_8b", 12288, 4096),
    ("small_k", 128, 4096),
    ("split_k", 8192, 256),
    ("tiny_n", 4096, 512),
];

fn bench_fp_gemv_typed<T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &MetalContext,
    label: &str,
) {
    let mut kernel = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(context, T::data_type())
        .expect("MatmulKernel");

    let mut group = c.benchmark_group(format!("{}/Kernel/GemvFp/{}", type_short_name::<Metal>(), label));

    for &(name, k, n) in ALL_SHAPES {
        let m = 1usize;
        let a = context
            .create_allocation(m * k * std::mem::size_of::<T>(), AllocationType::Global)
            .expect("a allocation");
        let b_array = context.create_array_uninitialized(&[n, k], T::data_type());
        let mut d = context
            .create_allocation(m * n * std::mem::size_of::<T>(), AllocationType::Global)
            .expect("d allocation");

        group.throughput(Throughput::Elements((m * n * k) as u64));
        group.bench_function(BenchmarkId::from_parameter(format!("{name}_K{k}N{n}")), |b| {
            iter_encode_loop::<Metal, _>(context, b, |encoder| {
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
                    .expect("encode failed");
            });
        });
    }
}

#[uzu_bench]
fn bench_fp_gemv(c: &mut Criterion) {
    let context = MetalContext::new().expect("Metal context");
    bench_fp_gemv_typed::<bf16>(c, &context, "BF16");
}
