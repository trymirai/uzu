use std::{cmp::Ordering, hint::black_box, time::Instant};

use half::bf16;
use num_traits::Float;
use proc_macros::uzu_test;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use super::unified_sampling;
use crate::{
    array::ArrayElement,
    encodable_block::sampling::{gumbel_float, revidx},
};

#[allow(clippy::too_many_arguments)]
fn reference_unified_sampling<T: ArrayElement + Float>(
    logits: *const T,
    output: *mut u32,
    seeds: Option<*const u64>,
    bitmask: Option<*const u32>,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    vocab_size: u32,
    batch_size: u32,
    is_stochastic: bool,
    has_bitmask: bool,
    has_temperature: bool,
    has_top_k: bool,
    has_top_p: bool,
    has_min_p: bool,
) {
    for batch_idx in 0..batch_size {
        let mut logits = unsafe {
            std::slice::from_raw_parts(logits.wrapping_add((vocab_size * batch_idx) as usize), vocab_size as usize)
        }
        .iter()
        .map(|logit| logit.to_f32().unwrap())
        .collect::<Vec<f32>>();

        if has_bitmask {
            let bitmask = unsafe {
                std::slice::from_raw_parts(
                    bitmask.unwrap().wrapping_add((vocab_size.div_ceil(u32::BITS) * batch_idx) as usize),
                    vocab_size.div_ceil(u32::BITS) as usize,
                )
            };
            for (logit_index, logit) in logits.iter_mut().enumerate() {
                if bitmask[logit_index / (u32::BITS as usize)] & (1 << (logit_index % (u32::BITS as usize))) == 0 {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }

        if has_temperature {
            let recip_temperature = 1.0 / temperature.unwrap();
            for logit in logits.iter_mut() {
                *logit *= recip_temperature;
            }
        }

        if has_top_k || has_top_p || has_min_p {
            let mut sorted_logits = logits.iter().copied().enumerate().collect::<Vec<_>>();
            sorted_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0)));

            let logits_max = sorted_logits[0].1;
            let logits_norm = sorted_logits.iter().map(|logit| (logit.1 - logits_max).exp()).sum::<f32>();

            logits.fill(f32::NEG_INFINITY);
            let mut top_p_mass = 0.0;
            for (top_k_num, (index, logit)) in sorted_logits.into_iter().enumerate() {
                if (has_top_k && top_k_num as u32 >= top_k.unwrap())
                    || (has_top_p && top_p_mass >= top_p.unwrap())
                    || (has_min_p && logit < logits_max + min_p.unwrap().ln())
                {
                    break;
                }
                logits[index] = logit;
                top_p_mass += (logit - logits_max).exp() / logits_norm;
            }
        }

        if is_stochastic {
            let seed = unsafe { *seeds.unwrap().wrapping_add(batch_idx as usize) };
            for (logit_index, logit) in logits.iter_mut().enumerate() {
                *logit += gumbel_float(seed, revidx(logit_index as u32, vocab_size));
            }
        }

        let argmax = logits
            .into_iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal).then(b.0.cmp(&a.0)))
            .unwrap()
            .0;

        unsafe { *output.wrapping_add(batch_idx as usize) = argmax as u32 }
    }
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Uniform,
    Peaked,
    Ties,
    HalfMasked,
}

fn make_logits(
    shape: Shape,
    vocab: usize,
    rng: &mut SmallRng,
) -> Vec<f32> {
    let mut logits = Vec::with_capacity(vocab);
    for index in 0..vocab {
        let value = match shape {
            Shape::Uniform => rng.random_range(-4.0f32..4.0),
            Shape::Peaked => {
                let base = rng.random_range(-12.0f32..-8.0);
                if index % 997 == 0 {
                    base + 20.0
                } else {
                    base
                }
            },
            Shape::Ties => ((rng.random_range(0u32..7) as f32) - 3.0) * 2.0,
            Shape::HalfMasked => {
                if index % 2 == 0 {
                    f32::NEG_INFINITY
                } else {
                    rng.random_range(-4.0f32..4.0)
                }
            },
        };
        logits.push(value);
    }
    logits
}

#[derive(Clone, Copy, Debug)]
struct Config {
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    stochastic: bool,
    bitmask: bool,
}

impl Config {
    fn sample(rng: &mut SmallRng) -> Self {
        Self {
            temperature: match rng.random_range(0u32..3) {
                0 => None,
                1 => Some(0.7),
                _ => Some(1.3),
            },
            top_k: match rng.random_range(0u32..5) {
                0 => None,
                1 => Some(1),
                2 => Some(5),
                3 => Some(40),
                _ => Some(1000),
            },
            top_p: match rng.random_range(0u32..5) {
                0 => None,
                1 => Some(0.1),
                2 => Some(0.5),
                3 => Some(0.9),
                _ => Some(0.999),
            },
            min_p: match rng.random_range(0u32..4) {
                0 => None,
                1 => Some(0.5),
                2 => Some(0.1),
                _ => Some(0.01),
            },
            stochastic: rng.random_range(0u32..2) == 0,
            bitmask: rng.random_range(0u32..4) == 0,
        }
    }
}

fn make_bitmask(
    vocab: usize,
    rng: &mut SmallRng,
) -> Vec<u32> {
    let words = (vocab as u32).div_ceil(u32::BITS) as usize;
    let mut bitmask = vec![0u32; words];
    for _ in 0..(vocab / 8).max(1) {
        let index = rng.random_range(0..vocab);
        bitmask[index / 32] |= 1 << (index % 32);
    }
    bitmask[0] |= 1;
    bitmask
}

fn run_both<T: ArrayElement + Float>(
    logits: &[T],
    bitmask: Option<&[u32]>,
    seeds: &[u64],
    config: Config,
    vocab: u32,
    batch: u32,
) -> (Vec<u32>, Vec<u32>) {
    let mut actual = vec![u32::MAX; batch as usize];
    let mut expected = vec![u32::MAX; batch as usize];

    let logits = logits.as_ptr();
    let seeds = config.stochastic.then_some(seeds.as_ptr());
    let bitmask_ptr = bitmask.map(|bitmask| bitmask.as_ptr());
    let has_bitmask = bitmask.is_some();
    let has_temperature = config.temperature.is_some();
    let has_top_k = config.top_k.is_some();
    let has_top_p = config.top_p.is_some();
    let has_min_p = config.min_p.is_some();

    unified_sampling::<T>(
        logits,
        actual.as_mut_ptr(),
        seeds,
        bitmask_ptr,
        config.temperature,
        config.top_k,
        config.top_p,
        config.min_p,
        vocab,
        batch,
        config.stochastic,
        has_bitmask,
        has_temperature,
        has_top_k,
        has_top_p,
        has_min_p,
    );
    reference_unified_sampling::<T>(
        logits,
        expected.as_mut_ptr(),
        seeds,
        bitmask_ptr,
        config.temperature,
        config.top_k,
        config.top_p,
        config.min_p,
        vocab,
        batch,
        config.stochastic,
        has_bitmask,
        has_temperature,
        has_top_k,
        has_top_p,
        has_min_p,
    );

    (actual, expected)
}

const SHAPES: [Shape; 4] = [Shape::Uniform, Shape::Peaked, Shape::Ties, Shape::HalfMasked];

#[uzu_test]
fn unified_sampling_matches_reference_f32() {
    let mut rng = SmallRng::seed_from_u64(0x9E3779B97F4A7C15);
    let mut cases = 0usize;

    for vocab in [17usize, 129, 1024, 4096] {
        for shape in SHAPES {
            for _ in 0..24 {
                let batch = 1 + rng.random_range(0u32..3);
                let mut logits = Vec::with_capacity(vocab * batch as usize);
                for _ in 0..batch {
                    logits.extend(make_logits(shape, vocab, &mut rng));
                }
                let config = Config::sample(&mut rng);
                let bitmask = config.bitmask.then(|| {
                    let mut bitmask = Vec::new();
                    for _ in 0..batch {
                        bitmask.extend(make_bitmask(vocab, &mut rng));
                    }
                    bitmask
                });
                let seeds: Vec<u64> = (0..batch).map(|_| rng.random::<u32>() as u64).collect();

                let (actual, expected) = run_both(&logits, bitmask.as_deref(), &seeds, config, vocab as u32, batch);
                assert_eq!(actual, expected, "vocab={vocab} shape={shape:?} batch={batch} config={config:?}");
                cases += 1;
            }
        }
    }

    assert!(cases >= 384, "expected a broad sweep, got {cases} cases");
}

#[uzu_test]
fn unified_sampling_matches_reference_bf16() {
    let mut rng = SmallRng::seed_from_u64(0xD1B54A32D192ED03);

    for vocab in [64usize, 512, 2048] {
        for shape in SHAPES {
            for _ in 0..12 {
                let logits: Vec<bf16> = make_logits(shape, vocab, &mut rng).into_iter().map(bf16::from_f32).collect();
                let config = Config::sample(&mut rng);
                let bitmask = config.bitmask.then(|| make_bitmask(vocab, &mut rng));
                let seeds = vec![rng.random::<u32>() as u64];

                let (actual, expected) = run_both(&logits, bitmask.as_deref(), &seeds, config, vocab as u32, 1);
                assert_eq!(actual, expected, "vocab={vocab} shape={shape:?} config={config:?}");
            }
        }
    }
}

#[uzu_test]
fn unified_sampling_matches_reference_full_vocab() {
    let vocab = 151_936usize;
    let mut rng = SmallRng::seed_from_u64(0x2545F4914F6CDD1D);
    let logits = make_logits(Shape::Peaked, vocab, &mut rng);

    for config in [
        Config {
            temperature: Some(0.7),
            top_k: Some(40),
            top_p: Some(0.95),
            min_p: None,
            stochastic: true,
            bitmask: false,
        },
        Config {
            temperature: None,
            top_k: None,
            top_p: Some(0.9),
            min_p: None,
            stochastic: true,
            bitmask: false,
        },
        Config {
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
            min_p: Some(0.05),
            stochastic: false,
            bitmask: false,
        },
        Config {
            temperature: None,
            top_k: Some(1),
            top_p: None,
            min_p: None,
            stochastic: true,
            bitmask: false,
        },
        Config {
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            stochastic: false,
            bitmask: false,
        },
    ] {
        let seeds = vec![0xABCD_EF01_2345_6789u64];
        let (actual, expected) = run_both(&logits, None, &seeds, config, vocab as u32, 1);
        assert_eq!(actual, expected, "config={config:?}");
    }
}

#[uzu_test]
fn unified_sampling_support_set_matches_reference() {
    let vocab = 2048usize;
    let mut rng = SmallRng::seed_from_u64(0x14057B7EF767814F);

    for shape in SHAPES {
        let logits = make_logits(shape, vocab, &mut rng);
        for config in [
            Config {
                temperature: None,
                top_k: Some(16),
                top_p: None,
                min_p: None,
                stochastic: true,
                bitmask: false,
            },
            Config {
                temperature: None,
                top_k: None,
                top_p: Some(0.9),
                min_p: None,
                stochastic: true,
                bitmask: false,
            },
            Config {
                temperature: Some(0.8),
                top_k: Some(64),
                top_p: Some(0.99),
                min_p: Some(0.02),
                stochastic: true,
                bitmask: false,
            },
        ] {
            let mut actual_support = std::collections::BTreeSet::new();
            let mut expected_support = std::collections::BTreeSet::new();
            for seed in 0..512u64 {
                let seeds = vec![seed.wrapping_mul(0x9E3779B97F4A7C15)];
                let (actual, expected) = run_both(&logits, None, &seeds, config, vocab as u32, 1);
                actual_support.insert(actual[0]);
                expected_support.insert(expected[0]);
            }
            assert_eq!(actual_support, expected_support, "shape={shape:?} config={config:?}");
            assert!(actual_support.len() > 1, "degenerate support for shape={shape:?} config={config:?}");
        }
    }
}

#[uzu_test]
fn unified_sampling_edge_cases_match_reference() {
    let greedy = Config {
        temperature: None,
        top_k: None,
        top_p: None,
        min_p: None,
        stochastic: false,
        bitmask: false,
    };

    let all_masked = vec![f32::NEG_INFINITY; 64];
    for config in [
        greedy,
        Config {
            top_k: Some(4),
            ..greedy
        },
        Config {
            top_p: Some(0.9),
            ..greedy
        },
        Config {
            min_p: Some(0.1),
            ..greedy
        },
    ] {
        let (actual, expected) = run_both(&all_masked, None, &[7], config, 64, 1);
        assert_eq!(actual, expected, "all -inf, config={config:?}");
    }

    let single = vec![1.5f32];
    let (actual, expected) = run_both(&single, None, &[3], greedy, 1, 1);
    assert_eq!(actual, expected);

    let flat = vec![2.0f32; 128];
    for config in [
        greedy,
        Config {
            top_k: Some(3),
            ..greedy
        },
        Config {
            top_p: Some(0.5),
            ..greedy
        },
    ] {
        let (actual, expected) = run_both(&flat, None, &[11], config, 128, 1);
        assert_eq!(actual, expected, "flat, config={config:?}");
        assert_eq!(actual[0], 0, "flat argmax must be the lowest index");
    }

    let mut rng = SmallRng::seed_from_u64(0x76E15D3EFEFDCBBF);
    let logits = make_logits(Shape::Peaked, 300, &mut rng);
    for config in [
        Config {
            top_k: Some(4096),
            ..greedy
        },
        Config {
            min_p: Some(1.0),
            ..greedy
        },
        Config {
            top_p: Some(0.0),
            ..greedy
        },
        Config {
            top_k: Some(1),
            top_p: Some(0.999),
            min_p: Some(0.001),
            ..greedy
        },
    ] {
        let (actual, expected) = run_both(&logits, None, &[5], config, 300, 1);
        assert_eq!(actual, expected, "config={config:?}");
    }
}

#[uzu_test]
fn unified_sampling_speedup_over_reference() {
    let vocab = 151_936usize;
    let mut rng = SmallRng::seed_from_u64(0x3C79AC492BA7B653);
    let logits = make_logits(Shape::Peaked, vocab, &mut rng);
    let config = Config {
        temperature: Some(0.7),
        top_k: Some(40),
        top_p: Some(0.95),
        min_p: None,
        stochastic: true,
        bitmask: false,
    };
    let seeds = vec![0x1234_5678_9ABC_DEF0u64];
    let mut output = vec![0u32; 1];
    let iterations = 20;

    let _ = run_both(&logits, None, &seeds, config, vocab as u32, 1);

    let started = Instant::now();
    for _ in 0..iterations {
        unified_sampling::<f32>(
            black_box(logits.as_ptr()),
            output.as_mut_ptr(),
            Some(seeds.as_ptr()),
            None,
            config.temperature,
            config.top_k,
            config.top_p,
            config.min_p,
            vocab as u32,
            1,
            true,
            false,
            true,
            true,
            true,
            false,
        );
    }
    let optimized = started.elapsed() / iterations;

    let started = Instant::now();
    for _ in 0..iterations {
        reference_unified_sampling::<f32>(
            black_box(logits.as_ptr()),
            output.as_mut_ptr(),
            Some(seeds.as_ptr()),
            None,
            config.temperature,
            config.top_k,
            config.top_p,
            config.min_p,
            vocab as u32,
            1,
            true,
            false,
            true,
            true,
            true,
            false,
        );
    }
    let baseline = started.elapsed() / iterations;

    println!(
        "unified_sampling vocab={vocab}: baseline {:?} -> optimized {:?} ({:.2}x)",
        baseline,
        optimized,
        baseline.as_secs_f64() / optimized.as_secs_f64()
    );
    assert!(optimized < baseline, "rewrite must not be slower: {optimized:?} vs {baseline:?}");
}
