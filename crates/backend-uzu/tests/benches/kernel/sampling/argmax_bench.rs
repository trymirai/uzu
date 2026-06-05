use std::mem::{MaybeUninit, size_of};

use backend_uzu::{
    array::{ArrayContextExt, ArrayElement},
    backends::common::{
        AllocationType, Backend, Context, Encoder, Kernels,
        gpu_types::ArgmaxPair,
        kernel::{ArgmaxFinalKernel, ArgmaxMainKernel, ArgmaxSingleKernel},
    },
    data_type::DataType,
};
use criterion::{BenchmarkId, Criterion, Throughput};
use num_traits::{Float, NumCast};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::Normal;

use crate::{common::type_short_name, uzu_bench};

fn get_argmax_data<T: ArrayElement + Float>(
    seed: u64,
    batch_size: usize,
    vocab_size: usize,
) -> Box<[T]> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let distribution = Normal::new(0.0, 32.0).unwrap();
    let mut logits: Box<[MaybeUninit<T>]> = Box::new_uninit_slice(batch_size * vocab_size);
    for logit in logits.iter_mut() {
        logit.write(<T as NumCast>::from(rng.sample(distribution)).unwrap());
    }
    unsafe { logits.assume_init() }
}

#[uzu_bench]
fn bench_argmax(c: &mut Criterion) {
    type T = f32;
    let batch_size = 1;

    for_each_backend!(|B| {
        let context = <B as Backend>::Context::new().unwrap();

        let single_kernel =
            <<B as Backend>::Kernels as Kernels>::ArgmaxSingleKernel::new(&context, T::data_type()).unwrap();
        let twopass_kernel_main =
            <<B as Backend>::Kernels as Kernels>::ArgmaxMainKernel::new(&context, T::data_type()).unwrap();
        let twopass_kernel_final = <<B as Backend>::Kernels as Kernels>::ArgmaxFinalKernel::new(&context).unwrap();

        let mut output_buffer = context.create_array_uninitialized(&[batch_size], DataType::U32).into_allocation();

        let mut group = c.benchmark_group(format!("{}/Kernel/Sampling/Argmax", type_short_name::<B>()));

        for vocab_size in [
            65536,  // LFM2-350M
            128256, // Llama-3.2-1B
            151936, // Qwen3-1.7B
            262144, // Gemma-3-1b
        ] {
            let logits_data = get_argmax_data::<T>(1337, batch_size, vocab_size);
            let logits_buffer = context.create_array_from(&[logits_data.len()], logits_data.as_ref()).into_allocation();
            let mut twopass_partial_results_buffer = context
                .create_allocation(
                    batch_size * vocab_size.div_ceil(4096) * size_of::<ArgmaxPair>(),
                    AllocationType::Global,
                )
                .unwrap();

            group.throughput(Throughput::Elements(vocab_size as u64));

            group.bench_function(BenchmarkId::new("SinglePass", format!("Vocab[{vocab_size}]")), |b| {
                b.iter_custom(|n_iters| {
                    let mut encoder = Encoder::<B>::new(&context).unwrap();

                    for _ in 0..n_iters {
                        single_kernel.encode(
                            &logits_buffer,
                            &mut output_buffer,
                            batch_size as u32,
                            vocab_size as u32,
                            &mut encoder,
                        );
                    }

                    encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
                })
            });

            group.bench_function(BenchmarkId::new("TwoPass", format!("Vocab[{vocab_size}]")), |b| {
                b.iter_custom(|n_iters| {
                    let mut encoder = Encoder::<B>::new(&context).unwrap();

                    for _ in 0..n_iters {
                        twopass_kernel_main.encode(
                            &logits_buffer,
                            &mut twopass_partial_results_buffer,
                            batch_size as u32,
                            vocab_size as u32,
                            &mut encoder,
                        );
                        twopass_kernel_final.encode(
                            &twopass_partial_results_buffer,
                            &mut output_buffer,
                            batch_size as u32,
                            vocab_size as u32,
                            &mut encoder,
                        );
                    }

                    encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
                })
            });
        }
    });
}
