use criterion::{BenchmarkId, Criterion, Throughput};
use itertools::iproduct;
use proc_macros::uzu_bench;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::{
    array::{ArrayContextExt, ArrayElement},
    backends::common::{Allocation, Backend, Context, Encoder, Kernels, kernel::RMSNormKernel},
    common::type_short_name,
};

fn get_rms_norm_data(
    seed: u64,
    batch_size: usize,
    model_dim: usize,
) -> (Box<[f32]>, Box<[f32]>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let input_size = batch_size * model_dim;
    let mut input_data = vec![0.0f32; input_size];
    for x in input_data.iter_mut() {
        *x = rng.random_range(-2.0f32..2.0f32);
    }
    let mut scale_data = vec![0.0f32; model_dim];
    for x in scale_data.iter_mut() {
        *x = rng.random_range(0.1f32..3.0f32);
    }
    (input_data.into_boxed_slice(), scale_data.into_boxed_slice())
}

#[uzu_bench]
fn bench_rms_norm(c: &mut Criterion) {
    type T = f32;
    let epsilon = 1e-6f32;
    let scale_offset = 0.0f32;

    for_each_backend!(|B| {
        let context = <B as Backend>::Context::new().unwrap();

        let kernel = <<B as Backend>::Kernels as Kernels>::RMSNormKernel::new(
            &context,
            T::data_type(),
            T::data_type(),
            T::data_type(),
            T::data_type(),
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        )
        .unwrap();

        let mut group = c.benchmark_group(format!("{}/Kernel/RMSNorm", type_short_name::<B>()));

        for (batch_size, model_dim) in iproduct!(
            [1, 4, 32, 128, 1024], // Batch sizes
            [
                1024, // LFM2-350M
                1152, // Gemma-3-1b
                2048, // Llama-3.2-1B
                2560, // Qwen3-4B
                3072, // Llama-3.2-3B
                4096, // Qwen3-8B
                5120, // Qwen2.5-32B
            ]
        ) {
            let (input_data, scale_data) = get_rms_norm_data(1337, batch_size, model_dim);
            let input_size = batch_size * model_dim;

            let input_buffer = context.create_array_from(&[input_size], input_data.as_ref()).into_allocation();
            let scales_buffer = context.create_array_from(&[model_dim], scale_data.as_ref()).into_allocation();
            let mut output_buffer = context.create_array_uninitialized(&[input_size], T::data_type()).into_allocation();

            group.throughput(Throughput::Elements((batch_size * model_dim) as u64));

            group.bench_function(BenchmarkId::from_parameter(format!("Batch[{batch_size}]Dim[{model_dim}]")), |b| {
                b.iter_custom(|n_iters| {
                    let mut encoder = Encoder::<B>::new(&context).unwrap();

                    for _ in 0..n_iters {
                        kernel.encode(
                            Some(&input_buffer),
                            &scales_buffer,
                            &mut output_buffer,
                            None::<&mut Allocation<B>>,
                            None::<&Allocation<B>>,
                            batch_size as u32,
                            model_dim as u32,
                            epsilon,
                            scale_offset,
                            1.0,
                            &mut encoder,
                        );
                    }

                    encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
                })
            });
        }
    });
}
