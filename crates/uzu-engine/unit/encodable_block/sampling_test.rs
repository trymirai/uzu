use std::{iter::repeat_with, mem::MaybeUninit};

use num_traits::{Float, NumCast};
use proptest::prelude::*;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::Normal;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        gpu_types::trie::TrieNode,
    },
    data_type::DataType,
    dispatch_dtype,
    encodable_block::{
        batch_topology::BatchTopology,
        sampling::{Sampling, SamplingMethod},
    },
    tests::proptest::{ComparableTestResults, TestContextes, for_each_context, kernel_data_type},
};
#[cfg(feature = "cpu")]
use crate::{backends::cpu::Cpu, tests::helpers::create_context};

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
    method: &SamplingMethod,
    batch_size: u32,
) -> Result<SamplingTestResults, TestCaseError> {
    let sampling = Sampling::<B>::new(T::data_type(), vocab_size as u32);

    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    let logits_buffer = command_buffer.allocate_constant_from_slice(logits).unwrap();
    let seeds_buffer = seeds.map(|seeds| command_buffer.allocate_constant_from_slice(seeds).unwrap());
    let bitmask_buffer = bitmask.map(|bitmask| command_buffer.allocate_constant_from_slice(bitmask).unwrap());

    let nodes = (0..batch_size)
        .map(|index| TrieNode {
            trie_start: index,
            trie_end: batch_size - 1,
            height: index,
        })
        .collect::<Box<[_]>>();
    let batch_topology = BatchTopology::new(&nodes, true);

    let sampled_buffer = sampling
        .encode(
            &logits_buffer,
            seeds_buffer.as_ref(),
            bitmask_buffer.as_ref(),
            None::<&B::GlobalBuffer>,   // TODO
            None::<&B::ConstantBuffer>, // TODO
            method,
            &batch_topology,
            (0..batch_size).into(),
            &mut command_buffer,
        )
        .unwrap();
    drop(logits_buffer);
    drop(seeds_buffer);
    drop(bitmask_buffer);
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    let sampled = sampled_buffer.copyout::<u32>();

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

            prop_oneof![
                Just(SamplingMethod::Greedy),
                (temperature, top_k, top_p, min_p).prop_map(|(temperature, top_k, top_p, min_p)| {
                    SamplingMethod::Stochastic {
                        temperature,
                        top_k,
                        top_p,
                        min_p,
                        repetition_penalty: None,       // TODO
                        suffix_repetition_length: None, // TODO
                    }
                },),
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
                &sampling_case.method,
                sampling_case.batch_size as u32,
            ))
            .compare_results()?
        });
    });
}

#[cfg(feature = "cpu")]
#[uzu_test]
#[ignore]
fn cpu_sampling_model_profile_witness() {
    // (V, temperature, top_k, top_p, min_p, repetition_penalty, suffix_repetition_length) from the model census; the
    // repetition penalty is a separate kernel, so it is recorded here but not passed to UnifiedSampling.
    let profiles = [
        (65536, 0.1, 50, 0.1, None, None, None),
        (128000, 0.1, 50, 0.1, None, None, None),
        (202048, 1.0, 64, 0.95, None, None, None),
        (248320, 0.6, 20, 0.95, None, Some(1.1), Some(1024)),
        (248320, 0.6, 20, 0.95, None, Some(1.05), Some(1024)),
        (248320, 0.6, 20, 0.95, None, None, None),
        (248320, 1.0, 20, 0.95, None, None, None),
    ];
    let context = create_context::<Cpu>();
    let mut cases = 0;
    let mut calls = 0;
    for (profile, parameters) in profiles.into_iter().enumerate() {
        let (vocab_size, temperature, top_k, top_p, min_p, repetition_penalty, suffix_repetition_length) = parameters;
        let method = SamplingMethod::Stochastic {
            temperature: Some(temperature),
            top_k: Some(top_k),
            top_p: Some(top_p),
            min_p,
            repetition_penalty: None,
            suffix_repetition_length: None,
        };
        for seed in [0, 1, u64::MAX] {
            for data_type in [DataType::F32, DataType::BF16] {
                let (sampling_seed, runs) = dispatch_dtype!(|(T: data_type)| {
                    let (logits, seeds, bitmask) = get_data::<T>(seed, 1, vocab_size, false, true);
                    assert!(logits.iter().all(|logit| logit.is_finite()));
                    let (seeds, bitmask) = (seeds.as_deref(), bitmask.as_deref());
                    let runs = [0, 1].map(|_| {
                        calls += 1;
                        do_sampling_backend::<Cpu, T>(&context, &logits, seeds, bitmask, vocab_size, &method, 1)
                            .unwrap()
                            .0
                    });
                    (seeds.unwrap()[0], runs)
                });
                cases += 1;
                assert_eq!(runs[0], runs[1], "profile {profile} seed {seed} {data_type:?}");
                assert!(runs[0].len() == 1 && runs[0][0] < vocab_size as u32);
                println!(
                    "UZU_SAMPLING_FINITE_RESULT {}",
                    serde_json::json!({
                        "profile": profile,
                        "vocab_size": vocab_size,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "min_p": min_p,
                        "repetition_penalty": repetition_penalty,
                        "suffix_repetition_length": suffix_repetition_length,
                        "dtype": format!("{data_type:?}"),
                        "seed": format!("{seed:#018x}"),
                        "sampling_seed": format!("{sampling_seed:#018x}"),
                        "runs": runs,
                    })
                );
            }
        }
    }
    println!(
        "UZU_SAMPLING_FINITE_END {}",
        serde_json::json!({ "profiles": profiles.len(), "seeds": 3, "dtypes": 2, "cases": cases, "calls": calls })
    );
    assert_eq!((profiles.len(), cases, calls), (7, 42, 84));
}
