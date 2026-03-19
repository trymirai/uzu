use std::mem::MaybeUninit;

use num_traits::{Float, NumCast};
use proptest::prelude::*;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::Normal;
use uzu::{
    ArrayElement,
    backends::common::{
        Backend, Buffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
        Context, Kernels,
        gpu_types::ArgmaxPair,
        kernel::{ArgmaxFinalKernel, ArgmaxMainKernel, ArgmaxSingleKernel},
    },
};

use crate::{
    common::proptest::{ComparableTestResults, TestContextes, kernel_data_type},
    dispatch_dtype, for_each_context,
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
    let logits_buffer = context.create_buffer(logits.len() * std::mem::size_of::<T>()).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut::<T>(logits_buffer.cpu_ptr().as_ptr() as *mut T, logits.len())
            .copy_from_slice(logits);
    }

    let mut twopass_partial_results_buffer =
        context.create_buffer(batch_size * vocab_size.div_ceil(4096) * std::mem::size_of::<ArgmaxPair>()).unwrap();

    let mut single_output_buffer = context.create_buffer(batch_size * 4).unwrap();
    let mut twopass_output_buffer = context.create_buffer(batch_size * 4).unwrap();

    let single_kernel = <B::Kernels as Kernels>::ArgmaxSingleKernel::new(context, T::data_type()).unwrap();
    let twopass_kernel_main = <B::Kernels as Kernels>::ArgmaxMainKernel::new(context, T::data_type()).unwrap();
    let twopass_kernel_final = <B::Kernels as Kernels>::ArgmaxFinalKernel::new(context).unwrap();

    let mut command_buffer = context.create_command_buffer().unwrap().start_encoding();
    single_kernel.encode(
        &logits_buffer,
        &mut single_output_buffer,
        batch_size as u32,
        vocab_size as u32,
        &mut command_buffer,
    );
    twopass_kernel_main.encode(
        &logits_buffer,
        &mut twopass_partial_results_buffer,
        batch_size as u32,
        vocab_size as u32,
        &mut command_buffer,
    );
    twopass_kernel_final.encode(
        &twopass_partial_results_buffer,
        &mut twopass_output_buffer,
        batch_size as u32,
        vocab_size as u32,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    let single_output = unsafe {
        std::slice::from_raw_parts(single_output_buffer.cpu_ptr().as_ptr() as *const u32, batch_size)
            .iter()
            .copied()
            .collect()
    };

    let twopass_output = unsafe {
        std::slice::from_raw_parts(twopass_output_buffer.cpu_ptr().as_ptr() as *const u32, batch_size)
            .iter()
            .copied()
            .collect()
    };

    prop_assert_eq!(&single_output, &twopass_output, "single and twopass argmax output differs");

    Ok(ArgmaxTestResults(single_output))
}

fn do_argmax<T: ArrayElement + Float>(
    contextes: &TestContextes,
    seed: u64,
    batch_size: usize,
    vocab_size: usize,
) -> Result<(), TestCaseError> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let distribution = Normal::new(0.0, 32.0).unwrap();
    let mut logits: Box<[MaybeUninit<T>]> = Box::new_uninit_slice(batch_size * vocab_size);
    for logit in logits.iter_mut() {
        logit.write(<T as NumCast>::from(rng.sample(distribution)).unwrap());
    }
    let logits = unsafe { logits.assume_init() };

    for_each_context!(contextes, |context: C| do_argmax_backend::<<C as Context>::Backend, T>(
        context,
        logits.as_ref(),
        batch_size,
        vocab_size
    ))
    .compare_results()
}

#[test]
fn test_argmax_prop() {
    let contextes = TestContextes::new();

    proptest!(|(data_type in kernel_data_type(), seed in any::<u64>().no_shrink(), batch_size in (1usize..=7usize), vocab_size in (1usize..=123456usize))| {
        dispatch_dtype!(|(T: data_type)| do_argmax::<T>(&contextes, seed, batch_size, vocab_size)?);
    });
}
