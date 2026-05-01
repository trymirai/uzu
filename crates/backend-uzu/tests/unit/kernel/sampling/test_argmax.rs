use std::mem::{MaybeUninit, size_of};

use criterion::{BenchmarkId, Criterion, Throughput};
use num_traits::{Float, NumCast};
use proptest::prelude::*;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::Normal;
use backend_uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::common::{
        AllocationType, Backend, Context, Encoder, Kernels,
        gpu_types::ArgmaxPair,
        kernel::{ArgmaxFinalKernel, ArgmaxMainKernel, ArgmaxSingleKernel},
    },
};

use crate::{
    common::{
        dispatch_dtype, for_each_context,
        proptest::{ComparableTestResults, TestContextes, kernel_data_type},
        type_short_name,
    },
    uzu_bench, uzu_test,
};

struct ArgmaxTestResults(Box<[u32]>);

impl ComparableTestResults for ArgmaxTestResults {
    fn compare(
        backend: &str,
        actual: &ArgmaxTestResults,
        reference: &ArgmaxTestResults,
    ) -> Result<(), TestCaseError> {
        prop_assert_eq!(&actual.0, &reference.0, "{} doesn't match cpu", backend);

        Ok(())
    }
}

fn do_argmax_backend<B: Backend, T: ArrayElement + Float>(
    context: &B::Context,
    logits: &[T],
    batch_size: usize,
    vocab_size: usize,
) -> Result<ArgmaxTestResults, TestCaseError> {
    let logits_buffer = context.create_array_from(&[logits.len()], logits, "").into_allocation();
    let mut twopass_partial_results_buffer = context
        .create_allocation(
            batch_size * vocab_size.div_ceil(4096) * size_of::<ArgmaxPair>(),
            AllocationType::Global,
        )
        .unwrap();
    let mut single_output_buffer = context.create_array_uninitialized(&[batch_size], DataType::U32, "").into_allocation();
    let mut twopass_output_buffer = context.create_array_uninitialized(&[batch_size], DataType::U32, "").into_allocation();

    let single_kernel = <B::Kernels as Kernels>::ArgmaxSingleKernel::new(context, T::data_type()).unwrap();
    let twopass_kernel_main = <B::Kernels as Kernels>::ArgmaxMainKernel::new(context, T::data_type()).unwrap();
    let twopass_kernel_final = <B::Kernels as Kernels>::ArgmaxFinalKernel::new(context).unwrap();

    let mut encoder = Encoder::new(context).unwrap();
    single_kernel.encode(&logits_buffer, &mut single_output_buffer, batch_size as u32, vocab_size as u32, &mut encoder);
    twopass_kernel_main.encode(
        &logits_buffer,
        &mut twopass_partial_results_buffer,
        batch_size as u32,
        vocab_size as u32,
        &mut encoder,
    );
    twopass_kernel_final.encode(
        &twopass_partial_results_buffer,
        &mut twopass_output_buffer,
        batch_size as u32,
        vocab_size as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let single_output = crate::common::helpers::allocation_to_vec::<B, u32>(&single_output_buffer);
    let twopass_output = crate::common::helpers::allocation_to_vec::<B, u32>(&twopass_output_buffer);

    prop_assert_eq!(&single_output, &twopass_output, "single and twopass argmax output differs");

    Ok(ArgmaxTestResults(single_output.into_boxed_slice()))
}

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

#[uzu_test]
fn test_argmax_prop() {
    let contextes = TestContextes::new();

    proptest!(|(data_type in kernel_data_type(), seed in any::<u64>().no_shrink(), batch_size in (1usize..=7usize), vocab_size in (1usize..=123456usize))| {
        dispatch_dtype!(|(T: data_type)| {
            let logits = get_argmax_data(seed, batch_size, vocab_size);

            for_each_context!(contextes, |context: C| do_argmax_backend::<<C as Context>::Backend, T>(
                context,
                logits.as_ref(),
                batch_size,
                vocab_size
            ))
            .compare_results()?
        });
    });
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

        let mut output_buffer = context.create_array_uninitialized(&[batch_size], DataType::U32, "").into_allocation();

        let mut group = c.benchmark_group(format!("{}/Kernel/Sampling/Argmax", type_short_name::<B>()));

        for vocab_size in [
            65536,  // LFM2-350M
            128256, // Llama-3.2-1B
            151936, // Qwen3-1.7B
            262144, // Gemma-3-1b
        ] {
            let logits_data = get_argmax_data::<T>(1337, batch_size, vocab_size);
            let logits_buffer = context.create_array_from(&[logits_data.len()], logits_data.as_ref(), "").into_allocation();
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
