#![cfg(metal_backend)]

use backend_uzu::{
    ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder,
            kernel::{ManualKernels, matmul::MatmulKernel},
        },
        metal::{Metal, MetalContext},
    },
};
use criterion::{BenchmarkId, Criterion};
use half::bf16;

use crate::{
    common::matmul::{SHAPES_BENCH, Variant, alloc_bench_buffers, encode_iteration},
    uzu_bench,
};

#[uzu_bench]
fn bench_matmul(criterion: &mut Criterion) {
    let context = MetalContext::new().unwrap();
    let mut kernel = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, bf16::data_type())
        .expect("MatmulKernel");

    for &variant in Variant::ALL {
        if !variant.supported(&context) {
            continue;
        }
        let mut group = criterion.benchmark_group(format!("Metal/Kernel/Matmul/{variant}"));

        for &shape in SHAPES_BENCH {
            let mut buffers = alloc_bench_buffers::<bf16>(&context, shape);
            let mut last_secs_per_iter: f64 = 0.0;
            group.bench_function(BenchmarkId::new("BF16", shape.to_string()), |bencher| {
                bencher.iter_custom(|iteration_count| {
                    let mut encoder = Encoder::<Metal>::new(&context).unwrap();
                    for _ in 0..iteration_count {
                        encode_iteration(&mut kernel, &mut buffers, shape, variant, &mut encoder);
                    }
                    let dur = encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time();
                    last_secs_per_iter = dur.as_secs_f64() / iteration_count as f64;
                    dur
                })
            });
            let gflops = shape.flops() as f64 / last_secs_per_iter / 1e9;
            eprintln!("  {variant} {shape}: {gflops:.2} GFLOPS");
        }
    }
}
