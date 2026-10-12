use std::{fmt::Debug, mem::size_of, ops::Range, time::Duration};

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{arg, cpu_buffer, cpu_submissions, kernel_fixture::KernelFixture};
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Kernels, kernel::UnifiedSamplingKernel},
        cpu::Cpu,
        vulkan::UnifiedSamplingVulkanKernel,
    },
    encodable_block::sampling::{gumbel_float, revidx},
    tests::helpers::{buffer_prefix_to_vec, create_context},
};

type Options = (bool, Option<f32>, Option<u32>, Option<f32>, Option<f32>);

fn cpu<T: ArrayElement + Float + Default>(
    logits: &[T],
    seeds: &[u64],
    mask: Option<&[u32]>,
    options: Options,
    vocab: u32,
    submissions: usize,
) -> (Vec<u32>, Vec<Duration>) {
    let context = create_context::<Cpu>();
    let (stochastic, temperature, top_k, top_p, min_p) = options;
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::UnifiedSamplingKernel::new(
        &context,
        T::data_type(),
        stochastic,
        mask.is_some(),
        temperature.is_some(),
        top_k.is_some(),
        top_p.is_some(),
        min_p.is_some(),
    )
    .expect("CPU UnifiedSampling");
    let input = cpu_buffer(&context, logits);
    let seed_buffer = cpu_buffer(&context, seeds);
    let mask_buffer = mask.map(|values| cpu_buffer(&context, values));
    let mut output = cpu_buffer(&context, &vec![u32::MAX; seeds.len()]);
    let times = cpu_submissions(&context, submissions, |encoding| {
        kernel.encode(
            &input,
            &mut output,
            stochastic.then_some(&seed_buffer),
            mask_buffer.as_ref(),
            temperature,
            top_k,
            top_p,
            min_p,
            vocab,
            seeds.len() as u32,
            encoding,
        );
    });
    (buffer_prefix_to_vec::<Cpu, u32>(&output, seeds.len()), times)
}

/// Metal's UnifiedSampling of one row of at most 1024 tokens, one per thread, in FP32 scalar steps. A logit is masked
/// to -inf, then scaled by 1 / temperature; NaN outranks nothing (Metal's Logit::operator>), ties go to the lower index
/// and signed zeros are equal; the candidate's own value is reloaded unmasked. Where Metal would load the token past
/// the row (no candidate is left), the Vulkan kernel's defined token 0. Sums run in token order; the thread reductions'
/// order only matters at an exact filter boundary, which these rows avoid.
fn metal_row(
    logits: &[f32],
    seed: u64,
    mask: Option<&[u32]>,
    (stochastic, temperature, top_k, top_p, min_p): Options,
) -> u32 {
    let vocab = logits.len();
    assert!(vocab <= 1024, "one token per thread");
    let scaled = |value: f32| temperature.map_or(value, |temperature| value * (1.0 / temperature));
    let value = |i: usize| match mask {
        Some(mask) if mask[i / 32] >> (i % 32) & 1 == 0 => scaled(f32::NEG_INFINITY),
        _ => scaled(logits[i]),
    };
    let score = |i: usize| match stochastic {
        true => value(i) + gumbel_float(seed, revidx(i as u32, vocab as u32)),
        false => value(i),
    };
    let outranks = |(a, i): (f32, usize), (b, j): (f32, usize)| a > b || (a == b && i < j);
    // The best score from Metal's Logit::LOWEST, which every non-NaN value outranks.
    let argmax = |tokens: &[usize]| {
        let lowest = (f32::NEG_INFINITY, usize::MAX);
        let best = tokens.iter().map(|&i| (score(i), i)).fold(lowest, |best, s| {
            if outranks(s, best) {
                s
            } else {
                best
            }
        });
        (best.1 != usize::MAX).then_some(best.1)
    };
    let maximum = (0..vocab).map(value).fold(f32::NEG_INFINITY, f32::max);
    // Each thread's running normalizer of its non -inf value, rescaled to the maximum unless zero.
    let norm = (0..vocab)
        .map(value)
        .filter(|&v| v != f32::NEG_INFINITY)
        .map(|v| {
            let local = f32::NEG_INFINITY.max(v);
            let norm = 0.0 * (f32::NEG_INFINITY - local).exp() + (v - local).exp();
            if norm != 0.0 {
                norm * (local - maximum).exp()
            } else {
                norm
            }
        })
        .sum::<f32>();
    let mut candidate = argmax(&(0..vocab).collect::<Vec<_>>());
    for iteration in 0..64 {
        let Some(token) = candidate else {
            return 0;
        };
        let pre = (scaled(logits[token]), token);
        let above = (0..vocab).filter(|&i| outranks((value(i), i), pre)).collect::<Vec<_>>();
        let mass = above.iter().map(|&i| (value(i) - maximum).exp() / norm).sum::<f32>();
        let passed = !top_k.is_some_and(|top_k| above.len() as u32 >= top_k)
            && !top_p.is_some_and(|top_p| mass >= top_p)
            && !min_p.is_some_and(|min_p| pre.0 < maximum + min_p.ln());
        if passed || iteration == 63 {
            return token as u32;
        }
        candidate = argmax(&above);
    }
    unreachable!()
}

/// `metal_row` of each row and its seed.
fn metal<T: ArrayElement + Float>(
    logits: &[T],
    seeds: &[u64],
    mask: Option<&[u32]>,
    options: Options,
    vocab: u32,
) -> Vec<u32> {
    let (vocab, words) = (vocab as usize, vocab.div_ceil(32) as usize);
    let logits = logits.iter().map(|value| value.to_f32().unwrap()).collect::<Vec<_>>();
    let rows = logits.chunks(vocab).zip(seeds).enumerate();
    rows.map(|(row, (logits, &seed))| metal_row(logits, seed, mask.map(|mask| &mask[row * words..][..words]), options))
        .collect()
}

/// Vulkan's tokens, which must be `expected`, with the read-only inputs and every guard unchanged.
fn check<T: ArrayElement + Float + Default + Debug>(
    fixture: &KernelFixture,
    logits: &[T],
    seeds: &[u64],
    mask: Option<&[u32]>,
    options: Options,
    vocab: u32,
    expected: &[u32],
) -> Vec<u32> {
    let sentinel = T::from(-31.0).unwrap();
    let mut input = fixture.guarded(&[vec![sentinel], logits.to_vec()].concat(), sentinel);
    input.1.start += size_of::<T>() as u64;
    let seed_buffer = fixture.guarded(seeds, u64::MAX);
    let mask_buffer = mask.map(|values| fixture.guarded(values, 0xa5a5_a5a5u32));
    let output = fixture.guarded(&vec![u32::MAX; seeds.len()], u32::MAX);
    let (stochastic, temperature, top_k, top_p, min_p) = options;
    let kernel = UnifiedSamplingVulkanKernel::new(
        &fixture.context,
        T::data_type(),
        stochastic,
        mask.is_some(),
        temperature.is_some(),
        top_k.is_some(),
        top_p.is_some(),
        min_p.is_some(),
    )
    .expect("Vulkan UnifiedSampling");
    let mut encoding = fixture.encoding();
    // SAFETY: disjoint guarded spans cover [rows,vocab], one seed per row and ceil(vocab/32) mask words per row.
    unsafe {
        kernel.encode(
            arg(&input),
            arg(&output),
            stochastic.then(|| arg(&seed_buffer)),
            mask_buffer.as_ref().map(arg),
            temperature,
            top_k,
            top_p,
            min_p,
            vocab,
            seeds.len() as u32,
            &mut encoding,
        );
    }
    KernelFixture::complete(encoding);
    // SAFETY: the submission completed before reading inputs, output and all guards.
    let actual = unsafe {
        KernelFixture::assert_unchanged(&input, sentinel, logits, "sampling logits");
        KernelFixture::assert_unchanged(&seed_buffer, u64::MAX, seeds, "sampling seeds");
        if let (Some(buffer), Some(values)) = (&mask_buffer, mask) {
            KernelFixture::assert_unchanged(buffer, 0xa5a5_a5a5u32, values, "sampling mask");
        }
        KernelFixture::read_guarded(&output, u32::MAX)
    };
    assert_eq!(actual, expected, "{:?} V={vocab} options={options:?} mask={}", T::data_type(), mask.is_some());
    actual
}

fn modes<T: ArrayElement + Float + Default + Debug>(fixture: &KernelFixture) {
    let seeds = [0, 1, u64::MAX];
    for vocab in [1u32, 31, 32, 33, 1023, 1024, 1025, 4095, 4096, 4097, 123456, 248320] {
        let logits = (0..3usize * vocab as usize)
            .map(|index| {
                let row = index / vocab as usize;
                let token = index % vocab as usize;
                let bits = (token as u32).wrapping_mul(747_796_405).wrapping_add(row as u32 * 289_133_645);
                T::from(((bits ^ (bits >> 16)) % 4096) as f32 / 256.0 - 8.0).unwrap()
            })
            .collect::<Vec<_>>();
        let mask = (0..3 * vocab.div_ceil(32)).map(|i| 0xb6db_6db6u32.rotate_left(i % 32)).collect::<Vec<_>>();
        let flags = if vocab == 4097 {
            (0..64).collect::<Vec<_>>()
        } else {
            vec![0, 1, 3, 7, 63]
        };
        for flags in flags {
            let mask = (flags & 2 != 0).then_some(mask.as_slice());
            let options = (
                flags & 1 != 0,
                (flags & 4 != 0).then_some(0.6),
                (flags & 8 != 0).then_some(20.min(vocab)),
                (flags & 16 != 0).then_some(0.95),
                (flags & 32 != 0).then_some(0.05),
            );
            let expected = cpu(&logits, &seeds, mask, options, vocab, 1).0;
            check(fixture, &logits, &seeds, mask, options, vocab, &expected);
        }
    }
}

#[uzu_test]
fn all_modes_and_vocabulary_boundaries() {
    let fixture = KernelFixture::new();
    modes::<f32>(&fixture);
    modes::<bf16>(&fixture);
    fixture.assert_clean();
}

/// Exceptional classes, ties and filter boundaries against Metal's design (`metal_row`), on which the CPU kernel differs
/// for NaN logits.
#[uzu_test]
fn ties_classes_and_filter_boundaries() {
    let fixture = KernelFixture::new();
    let rows = [
        [0.0, -0.0, 0.0, -0.0],
        [f32::NAN, -1.0, 2.0, f32::NAN],
        [f32::NAN; 4],
        [f32::NEG_INFINITY; 4],
        [f32::INFINITY, f32::INFINITY, 0.0, f32::NEG_INFINITY],
        [f32::from_bits(1), -f32::from_bits(1), 0.0, -0.0],
        [0.0, -90.0, -100.0, -1000.0],
    ];
    for row in rows {
        for stochastic in [false, true] {
            for (temperature, top_k, top_p, min_p) in [
                (None, None, None, None),
                (None, Some(1), None, None),
                (None, Some(0), None, None),
                (None, None, Some(0.0), None),
                (None, None, Some(0.5), None),
                (None, None, Some(f32::NAN), None),
                (None, None, None, Some(0.0)),
                (None, None, None, Some(f32::from_bits(1))),
                (None, None, None, Some(-1.0)),
                (None, None, None, Some(f32::NAN)),
                (None, None, None, Some(2.0)),
                (Some(0.0), None, None, None),
                (Some(-1.0), None, None, None),
                (Some(f32::NAN), None, None, None),
            ] {
                let options = (stochastic, temperature, top_k, top_p, min_p);
                let bf16_row = row.map(bf16::from_f32);
                check(&fixture, &row, &[u64::MAX], None, options, 4, &metal(&row, &[u64::MAX], None, options, 4));
                let expected = metal(&bf16_row, &[u64::MAX], None, options, 4);
                check(&fixture, &bf16_row, &[u64::MAX], None, options, 4, &expected);
            }
        }
    }
    for mask in [[0u32], [2], [8], [15]] {
        let expected = [match mask[0] {
            0 => 0,
            8 => 3,
            _ => 1,
        }];
        let (logits, options) = ([1.0, 4.0, 3.0, 2.0], (false, None, None, None, None));
        assert_eq!(metal::<f32>(&logits, &[0], Some(&mask), options, 4), expected);
        check::<f32>(&fixture, &logits, &[0], Some(&mask), options, 4, &expected);
    }
    let (logits, options) = ([0.0, -0.0, 0.0, -0.0], (false, None, Some(2), Some(0.25), None));
    assert_eq!(metal::<f32>(&logits, &[0], None, options, 4), [0]);
    check::<f32>(&fixture, &logits, &[0], None, options, 4, &[0]);
    fixture.assert_clean();
}

/// NaN beside -inf, signed zeros and a 1000 logit across mask-word boundaries (V 21, 32 and 33), all-NaN rows, NaN
/// between infinities, masked tokens between NaNs, zero temperature and signed zeros, each under every top_k 1, top_p
/// 0.9 and min_p 0.5 combination, plus an empty top_k and top_p, F32 and BF16, deterministic and with seeds 0, 1 and
/// u64::MAX: Vulkan gives Metal's token (`metal_row`), which lies in the hand-derived `tokens` (1000-logit gaps exceed
/// the finite Gumbel range).
#[uzu_test]
fn nan_rows_and_mask_boundaries() {
    let fixture = KernelFixture::new();
    let regression_row = |vocab_size: usize| {
        let mut logits = vec![-1000.0; vocab_size];
        logits[..4].copy_from_slice(&[f32::NAN, f32::NEG_INFINITY, -0.0, 0.0]);
        logits[vocab_size / 2] = f32::NAN;
        logits[vocab_size - 1] = 1000.0;
        logits
    };
    let rows: [(&str, Vec<f32>, Option<u32>, Option<f32>, Range<u32>); 9] = [
        ("nan_v21", regression_row(21), None, None, 20..21),
        ("nan_v32", regression_row(32), None, None, 31..32),
        ("nan_v33", regression_row(33), None, None, 32..33),
        ("all_nan", vec![f32::NAN; 33], None, None, 0..1),
        ("nan_first", vec![f32::NAN, -1000.0, 1000.0, -1000.0], None, None, 2..3),
        ("nan_between_infinities", vec![f32::NEG_INFINITY, f32::NAN, f32::INFINITY], None, None, 2..3),
        ("masked_between_nans", vec![f32::NAN, f32::NEG_INFINITY, f32::NAN], Some(0b101), None, 1..2),
        ("zero_temperature", vec![0.0, -1.0, 2.0], None, Some(0.0), 2..3),
        ("signed_zeros", vec![-1000.0, -0.0, 0.0], None, None, 1..3),
    ];
    let mut cases = rows
        .into_iter()
        .flat_map(|(label, logits, mask, temperature, tokens)| {
            (0..8).map(move |filters| {
                let filters = (
                    (filters & 1 != 0).then_some(1),
                    (filters & 2 != 0).then_some(0.9),
                    (filters & 4 != 0).then_some(0.5),
                );
                (label, logits.clone(), mask, temperature, filters, tokens.clone())
            })
        })
        .collect::<Vec<_>>();
    cases.extend([
        ("empty_top_k", vec![-1000.0, 1000.0], None, None, (Some(0), None, None), 0..1),
        ("empty_top_p", vec![-1000.0, 1000.0], None, None, (None, Some(0.0), None), 0..1),
    ]);
    for (label, logits, mask, temperature, (top_k, top_p, min_p), tokens) in cases {
        let vocab = logits.len() as u32;
        for (stochastic, seeds) in [(false, &[0][..]), (true, &[0, 1, u64::MAX][..])] {
            let options = (stochastic, temperature, top_k, top_p, min_p);
            let logits = logits.repeat(seeds.len());
            let mask = mask.map(|mask| vec![mask; seeds.len()]);
            let case = format!("{label} stochastic={stochastic} top_k={top_k:?} top_p={top_p:?} min_p={min_p:?}");
            let expected = metal(&logits, seeds, mask.as_deref(), options, vocab);
            assert!(expected.iter().all(|token| tokens.contains(token)), "{case}: Metal gives {expected:?}");
            check(&fixture, &logits, seeds, mask.as_deref(), options, vocab, &expected);
            let logits = logits.iter().copied().map(bf16::from_f32).collect::<Vec<_>>();
            assert_eq!(metal(&logits, seeds, mask.as_deref(), options, vocab), expected, "{case}: BF16");
            check(&fixture, &logits, seeds, mask.as_deref(), options, vocab, &expected);
        }
    }
    fixture.assert_clean();
}

#[uzu_test]
fn batches_and_zero_work() {
    let fixture = KernelFixture::new();
    let rows = 257;
    check(&fixture, &vec![1.0f32; rows], &vec![0; rows], None, (false, None, None, None, None), 1, &vec![0; rows]);
    check(&fixture, &[] as &[f32], &[], None, (false, None, None, None, None), 1, &[]);
    fixture.assert_clean();
}

#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    for vocab in [65536u32, 128000, 202048, 248320] {
        for rows in [1, 8] {
            let seeds = vec![u64::MAX; rows];
            let logits = (0..vocab as usize * rows).map(|i| (i % 8192) as f32 / 512.0 - 8.0).collect::<Vec<_>>();
            for options in [
                (false, None, None, None, None),
                (true, None, None, None, None),
                (true, Some(0.6), Some(20), Some(0.95), None),
            ] {
                let (expected, mut times) = cpu(&logits, &seeds, None, options, vocab, 13);
                check(&fixture, &logits, &seeds, None, options, vocab, &expected);
                times.drain(..3);
                times.sort();
                let input = fixture.guarded(&logits, -31.0f32);
                let seed_buffer = fixture.guarded(&seeds, u64::MAX);
                let output = fixture.guarded(&vec![u32::MAX; rows], u32::MAX);
                let (stochastic, temperature, top_k, top_p, min_p) = options;
                let kernel = UnifiedSamplingVulkanKernel::new(
                    &fixture.context,
                    crate::data_type::DataType::F32,
                    stochastic,
                    false,
                    temperature.is_some(),
                    top_k.is_some(),
                    top_p.is_some(),
                    false,
                )
                .unwrap();
                // SAFETY: the same dense spans and flags passed the guarded correctness submission above.
                let (gpu, wall) = fixture.median_times(|encoding| unsafe {
                    kernel.encode(
                        arg(&input),
                        arg(&output),
                        stochastic.then(|| arg(&seed_buffer)),
                        None,
                        temperature,
                        top_k,
                        top_p,
                        min_p,
                        vocab,
                        rows as u32,
                        encoding,
                    );
                });
                // SAFETY: every timed submission completed; output guards remain checked.
                assert_eq!(unsafe { KernelFixture::read_guarded(&output, u32::MAX) }, expected);
                eprintln!(
                    "UnifiedSampling F32 V={vocab} rows={rows} options={options:?}: GPU {gpu:?}, wall {wall:?}, CPU wall {:?}",
                    times[times.len() / 2]
                );
            }
        }
    }
    fixture.assert_clean();
}
