use std::{iter::repeat_with, mem::MaybeUninit};

use backend_uzu::{
    array::ArrayElement,
    backends::common::{AllocationType, Backend, Context, Encoder},
    dispatch_dtype,
};
use num_traits::{Float, NumCast};
use proc_macros::uzu_test;
use proptest::prelude::*;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::Normal;

use crate::{
    data_type::DataType,
    encodable_block::Sampling,
    session::parameter::{SamplingMethod, SamplingProcessingOrder},
    tests::proptest::{ComparableTestResults, TestContextes, for_each_context, kernel_data_type},
};

struct SamplingTestResults(Vec<u32>);

impl ComparableTestResults for SamplingTestResults {
    fn compare(
        backend: &str,
        actual: &SamplingTestResults,
        reference: &SamplingTestResults,
    ) -> Result<(), TestCaseError> {
        prop_assert_eq!(&actual.0, &reference.0, "{} doesn't match cpu", backend);

        Ok(())
    }
}

fn do_sampling_backend<B: Backend, T: ArrayElement + Float>(
    context: &B::Context,
    logits: &[T],
    seeds: Option<&[u64]>,
    bitmask: Option<&[u32]>,
    vocab_size: usize,
    method: SamplingMethod,
    batch_size: usize,
) -> Result<SamplingTestResults, TestCaseError> {
    let sampling = Sampling::new(T::data_type(), vocab_size);

    let mut logits_allocation =
        context.create_allocation(logits.len() * T::data_type().size_in_bytes(), AllocationType::Global).unwrap();
    logits_allocation.copyin(logits);
    let seeds_allocation = if let Some(seeds) = seeds {
        let mut seeds_allocation = context.create_allocation(seeds.len() * 8, AllocationType::Global).unwrap();
        seeds_allocation.copyin(seeds);
        Some(seeds_allocation)
    } else {
        None
    };
    let bitmask_allocation = if let Some(bitmask) = bitmask {
        let mut bitmask_allocation = context.create_allocation(bitmask.len() * 4, AllocationType::Global).unwrap();
        bitmask_allocation.copyin(bitmask);
        Some(bitmask_allocation)
    } else {
        None
    };

    let mut encoder = Encoder::new(context).unwrap();
    let sampled_allocation = sampling
        .encode(
            &logits_allocation,
            seeds_allocation.as_ref(),
            bitmask_allocation.as_ref(),
            None, // TODO
            None, // TODO
            method,
            batch_size,
            &mut encoder,
        )
        .unwrap();
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let sampled = sampled_allocation.copyout::<u32>();

    Ok(SamplingTestResults(sampled))
}

fn get_data<T: ArrayElement + Float>(
    seed: u64,
    batch_size: usize,
    vocab_size: usize,
    bitmask: bool,
    stochastic: bool,
) -> (Box<[T]>, Option<Box<[u64]>>, Option<Box<[u32]>>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mean = rng.random_range(-100.0..100.0);
    let std_dev = rng.random_range(24.0..64.0);

    let distribution: Normal<f32> = Normal::new(mean, std_dev).unwrap();
    let mut logits: Box<[MaybeUninit<T>]> = Box::new_uninit_slice(batch_size * vocab_size);
    for logit in logits.iter_mut() {
        logit.write(<T as NumCast>::from(rng.sample(distribution)).unwrap());
    }

    let seeds = stochastic.then(|| repeat_with(|| rng.random::<u64>()).take(batch_size).collect());
    let bitmask = bitmask.then(|| {
        let bitmask_size = vocab_size.div_ceil(u32::BITS as usize);
        let mut bitmask: Box<[MaybeUninit<u32>]> = Box::new_uninit_slice(batch_size * bitmask_size);
        for entry in bitmask.iter_mut() {
            entry.write(rng.random::<u32>());
        }
        unsafe { bitmask.assume_init() }
    });

    (unsafe { logits.assume_init() }, seeds, bitmask)
}

#[derive(Debug, Clone)]
struct SamplingCase {
    data_type: DataType,
    seed: u64,
    batch_size: usize,
    vocab_size: usize,
    bitmask: bool,
    method: SamplingMethod,
}

fn sampling_case() -> impl Strategy<Value = SamplingCase> {
    (kernel_data_type(), any::<u64>().no_shrink(), 1usize..=7, 1usize..=123456, any::<bool>()).prop_flat_map(
        |(data_type, seed, batch_size, vocab_size, bitmask)| {
            let temperature = prop_oneof![Just(None), (0.05f32..=4.0).prop_map(Some),];

            let top_k = prop_oneof![Just(None), (1u32..=vocab_size as u32).prop_map(Some),];

            let top_p = prop_oneof![Just(None), (0.01f32..=1.0).prop_map(Some),];

            let min_p = prop_oneof![Just(None), (0.0f32..=1.0).prop_map(Some),];

            let order = prop_oneof![
                Just(SamplingProcessingOrder::TemperatureThenFilters),
                Just(SamplingProcessingOrder::FiltersThenTemperature),
            ];

            prop_oneof![
                Just(SamplingMethod::Greedy),
                (temperature, top_k, top_p, min_p, order).prop_map(
                    |(temperature, top_k, top_p, min_p, processing_order)| {
                        SamplingMethod::Stochastic {
                            temperature,
                            top_k,
                            top_p,
                            min_p,
                            repetition_penalty: None,       // TODO
                            suffix_repetition_length: None, // TODO
                            processing_order,
                        }
                    },
                ),
            ]
            .prop_map(move |method| SamplingCase {
                data_type,
                seed,
                batch_size,
                vocab_size,
                bitmask,
                method,
            })
        },
    )
}

#[uzu_test]
fn test_sampling_prop() {
    let contextes = TestContextes::new();

    proptest!(|(sampling_case in sampling_case())| {
        dispatch_dtype!(|(T: sampling_case.data_type)| {
            let (logits, seeds, bitmask) = get_data(sampling_case.seed, sampling_case.batch_size, sampling_case.vocab_size, sampling_case.bitmask, matches!(sampling_case.method, SamplingMethod::Stochastic { .. }));

            for_each_context!(contextes, |context: C| do_sampling_backend::<<C as Context>::Backend, T>(
                context,
                logits.as_ref(),
                seeds.as_ref().map(Box::as_ref),
                bitmask.as_ref().map(Box::as_ref),
                sampling_case.vocab_size,
                sampling_case.method,
                sampling_case.batch_size,
            ))
            .compare_results()?
        });
    });
}
