use std::time::Duration;

use backend_uzu::{
    array::{ArrayContextExt, ArrayElement},
    backends::{
        common::{Allocation, Backend, Context, Encoder, Kernels, kernel::RMSNormKernel},
        cpu::Cpu,
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::common::matmul::iter_encode_loop_named;

const RMS_NORM_BATCH_SIZES: &[usize] = &[1, 4, 32, 128, 512];
const RMS_NORM_MODEL_DIMS: &[usize] = &[1024, 2048, 3072, 4096, 8192];

fn rms_norm_data(
    seed: u64,
    batch_size: usize,
    model_dim: usize,
) -> (Box<[bf16]>, Box<[bf16]>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let input = (0..batch_size * model_dim)
        .map(|_| bf16::from_f32(rng.random_range(-2.0f32..2.0f32)))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let scales =
        (0..model_dim).map(|_| bf16::from_f32(rng.random_range(0.1f32..3.0f32))).collect::<Vec<_>>().into_boxed_slice();
    (input, scales)
}

#[uzu_bench]
fn bench_cpu_rms_norm(c: &mut Criterion) {
    let context = <Cpu as Backend>::Context::new().expect("CPU context");
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::RMSNormKernel::new(
        &context,
        bf16::data_type(),
        bf16::data_type(),
        bf16::data_type(),
        f32::data_type(),
        false,
        false,
        false,
        false,
        false,
        false,
        false,
    )
    .expect("CPU RMSNormKernel");

    let group_path = "Cpu/Kernel/RMSNorm";
    let mut group = c.benchmark_group(group_path);
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for &batch_size in RMS_NORM_BATCH_SIZES {
        for &model_dim in RMS_NORM_MODEL_DIMS {
            let input_size = batch_size * model_dim;
            let (input_data, scale_data) = rms_norm_data(1337, batch_size, model_dim);

            let input = context.create_array_from(&[input_size], input_data.as_ref()).into_allocation();
            let scales = context.create_array_from(&[model_dim], scale_data.as_ref()).into_allocation();
            let mut output = context.create_array_uninitialized(&[input_size], bf16::data_type()).into_allocation();

            group.throughput(Throughput::Elements(input_size as u64));
            group.bench_function(BenchmarkId::from_parameter(format!("Batch[{batch_size}]Dim[{model_dim}]")), |b| {
                let benchmark_path = format!("{group_path}/Batch[{batch_size}]Dim[{model_dim}]");
                iter_encode_loop_named::<Cpu, _>(&context, b, &benchmark_path, |encoder: &mut Encoder<Cpu>| {
                    kernel.encode(
                        Some(&input),
                        &scales,
                        &mut output,
                        None::<&mut Allocation<Cpu>>,
                        None::<&Allocation<Cpu>>,
                        batch_size as u32,
                        model_dim as u32,
                        1e-6,
                        0.0,
                        1.0,
                        encoder,
                    );
                });
            });
        }
    }

    group.finish();
}
