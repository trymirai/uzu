use std::mem::{MaybeUninit, size_of};

use num_traits::{Float, NumCast};
use proptest::prelude::*;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::Normal;
use backend_uzu::{
    ArrayContextExt, ArrayElement, DataType, dispatch_dtype,
    backends::common::{
        AllocationType, Backend, Context, Encoder, Kernels,
        gpu_types::ArgmaxPair,
        kernel::{ArgmaxFinalKernel, ArgmaxMainKernel, ArgmaxSingleKernel},
    },
};

use crate::{
    common::{
        for_each_context,
        proptest::{ComparableTestResults, TestContextes, kernel_data_type},
    },
    uzu_test,
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
    let logits_buffer = context.create_array_from(&[logits.len()], logits).into_allocation();
    let mut twopass_partial_results_buffer = context
        .create_allocation(
            batch_size * vocab_size.div_ceil(4096) * size_of::<ArgmaxPair>(),
            AllocationType::Global,
        )
        .unwrap();
    let mut single_output_buffer = context.create_array_uninitialized(&[batch_size], DataType::U32).into_allocation();
    let mut twopass_output_buffer = context.create_array_uninitialized(&[batch_size], DataType::U32).into_allocation();

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
