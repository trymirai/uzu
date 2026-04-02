use std::{
    collections::HashSet,
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::{TopPKernel, sampling::SamplingKernel},
        },
        cpu::Cpu,
    },
    session::parameter::{SamplingMethod, SamplingProcessingOrder},
};

use crate::uzu_test;

const TEST_SAMPLING_SEED: u64 = 42;

struct Input<T: ArrayElement + Float> {
    logits: Box<[T]>,
    batch_size: u32,
    vocab_size: u32,
    top_p: f32,
    in_place: bool,
}

fn get_test_data<T: ArrayElement + Float>(
    batch_size: u32,
    vocab_size: u32,
    top_p: f32,
    in_place: bool,
) -> (Input<T>, Vec<T>) {
    let len = (batch_size * vocab_size) as usize;
    let mut rng = StdRng::seed_from_u64(42);
    let mut logits: Vec<T> = vec![T::zero(); len];
    for x in logits.iter_mut() {
        *x = T::from(rng.random_range(-16.0f32..16.0f32)).unwrap();
    }

    let input = Input {
        logits: logits.into_boxed_slice(),
        batch_size,
        vocab_size,
        top_p,
        in_place,
    };

    let expected = get_output::<T, Cpu>(&input);

    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = <B as Backend>::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::TopPKernel::new(&context, T::data_type(), input.in_place)
        .expect("Failed to create TopPKernel");

    let len = (input.batch_size * input.vocab_size) as usize;
    let logits_array = context.create_array_from(&[len], &input.logits, "");
    let logits_array_buffer_rc = logits_array.buffer();
    let logits_array_borrow = logits_array_buffer_rc.borrow();
    let logits_array_deref = logits_array_borrow.deref();
    let logits_buffer = (!input.in_place).then(|| logits_array_deref);
    let output_array = match input.in_place {
        true => context.create_array_from(&[len], &input.logits, ""),
        false => context.create_array_uninitialized(&[len], T::data_type(), ""),
    };

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        logits_buffer,
        output_array.buffer().borrow_mut().deref_mut(),
        input.batch_size,
        input.vocab_size,
        input.top_p,
        &mut encoder,
    );

    encoder.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice().to_vec()
}

fn assert_top_p_equal<T: ArrayElement + Float + Display>(
    expected: &[T],
    actual: &[T],
    vocab_size: u32,
    eps: f32,
    msg: &str,
) {
    assert_eq!(expected.len(), actual.len(), "Slices size mismatch");
    let vocab_size = vocab_size as usize;
    let num_batches = expected.len() / vocab_size;

    for batch_idx in 0..num_batches {
        let start = batch_idx * vocab_size;
        let end = start + vocab_size;

        let expected_kept: usize = expected[start..end].iter().filter(|v| v.is_finite()).count();
        let actual_kept: usize = actual[start..end].iter().filter(|v| v.is_finite()).count();

        // Allow small differences due to binary search convergence
        let diff = (expected_kept as i64 - actual_kept as i64).unsigned_abs();
        assert!(diff <= 3, "{msg}. Batch {batch_idx}: expected {expected_kept} kept values, got {actual_kept}");

        // For values that are kept by both, they should match the original values
        for i in start..end {
            let e = expected[i].to_f32().unwrap();
            let a = actual[i].to_f32().unwrap();
            if e.is_finite() && a.is_finite() {
                let d = (e - a).abs();
                assert!(d < eps, "{msg}. Mismatch at index {i}: expected {e}, got {a}, diff {d}");
            }
        }
    }
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    batch_size: u32,
    vocab_size: u32,
    top_p: f32,
    in_place: bool,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2
    } else {
        1e-5
    };

    let (input, expected) = get_test_data::<T>(batch_size, vocab_size, top_p, in_place);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        let msg = format!("Results are not equal for backend {}", std::any::type_name::<B>());
        assert_top_p_equal::<T>(&expected, &output, vocab_size, eps, &msg);
    });
}

// Out-of-place tests (same data as sampling_test::test_topp_gpu_cpu_match)
#[uzu_test]
fn test_f32() {
    test_internal::<f32>(4, 1024, 0.9, false);
}

#[uzu_test]
fn test_f16() {
    test_internal::<f16>(4, 1024, 0.9, false);
}

#[uzu_test]
fn test_bf16() {
    test_internal::<bf16>(4, 1024, 0.9, false);
}

// In-place tests
#[uzu_test]
fn test_in_place_f32() {
    test_internal::<f32>(4, 1024, 0.9, true);
}

#[uzu_test]
fn test_in_place_f16() {
    test_internal::<f16>(4, 1024, 0.9, true);
}

#[uzu_test]
fn test_in_place_bf16() {
    test_internal::<bf16>(4, 1024, 0.9, true);
}

// Edge cases
#[uzu_test]
fn test_single_batch_f32() {
    test_internal::<f32>(1, 1024, 0.9, false);
}

#[uzu_test]
fn test_top_p_very_small() {
    test_internal::<f32>(4, 1024, 0.01, false);
}

#[uzu_test]
fn test_top_p_near_1() {
    test_internal::<f32>(4, 1024, 0.99, false);
}

// --- Sampling-level top-p tests (test the full SamplingKernel with top_p) ---

fn cpu_reference_top_p(
    row_logits: &[f32],
    top_p: f32,
) -> Vec<f32> {
    let max_logit = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = row_logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum_exp).collect();

    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cum = 0.0f32;
    let mut keep = vec![false; row_logits.len()];
    for &(idx, p) in &indexed {
        if cum >= top_p {
            break;
        }
        keep[idx] = true;
        cum += p;
    }

    // Renormalise kept probabilities
    let kept_sum: f32 = probs.iter().zip(keep.iter()).filter(|&(_, &k)| k).map(|(&p, _)| p).sum();
    probs
        .iter()
        .zip(keep.iter())
        .map(|(&p, &k)| {
            if k {
                p / kept_sum
            } else {
                0.0
            }
        })
        .collect()
}

fn test_topp_sampling_from_prob_exact_match_internal<B: Backend>(
    batch_size: usize,
    k: usize,
    vocab_size: usize,
) {
    let context = <B as Backend>::Context::new().expect("Failed to create Context");

    let p = (k as f32) * 0.1;

    let kernel = SamplingKernel::<B>::new(&context, DataType::F32, batch_size, vocab_size)
        .expect("Failed to create sampling kernel");

    // Build probability table
    let low_prob = (1.0 - p) / (vocab_size - k) as f32;
    let mut probs = vec![low_prob; batch_size * vocab_size];
    let mut high_prob_token_sets: Vec<HashSet<usize>> = Vec::with_capacity(batch_size);

    let mut rng = StdRng::seed_from_u64(42);
    let mut all_token_ids: Vec<usize> = (0..vocab_size).collect();

    for b in 0..batch_size {
        all_token_ids.shuffle(&mut rng);
        let high_prob_tokens: HashSet<usize> = all_token_ids[..k].iter().cloned().collect();

        for &token_id in &high_prob_tokens {
            probs[b * vocab_size + token_id] = 0.1;
        }
        high_prob_token_sets.push(high_prob_tokens);
    }

    let logits: Vec<f32> = probs
        .iter()
        .map(|&prob| {
            if prob <= 0.0 {
                -50.0
            } else {
                prob.ln()
            }
        })
        .collect();

    let output_array = context.create_array_uninitialized(&[batch_size], DataType::U32, "");

    let num_samples = 1000;
    let mut counter = vec![0i32; batch_size * vocab_size];

    for draw in 0..num_samples {
        let logits_array = context.create_array_from(&[batch_size * vocab_size], &logits, "");
        let seeds: Vec<u64> = vec![TEST_SAMPLING_SEED + draw as u64; batch_size];
        let seeds_array = context.create_array_from(&[batch_size], &seeds, "");

        let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
        kernel
            .encode(
                logits_array.buffer().borrow_mut().deref_mut(),
                seeds_array.buffer().borrow().deref(),
                0,
                None,
                0,
                output_array.buffer().borrow_mut().deref_mut(),
                SamplingMethod::Stochastic {
                    temperature: None,
                    top_k: None,
                    top_p: Some(p),
                    min_p: None,
                    processing_order: SamplingProcessingOrder::TemperatureThenFilters,
                },
                batch_size,
                vocab_size,
                &mut encoder,
            )
            .expect("encode");
        encoder.end_encoding().submit().wait_until_completed().unwrap();

        let sampled_ids: &[u32] = output_array.as_slice();

        for (i, &sampled_id) in sampled_ids.iter().enumerate() {
            assert!((sampled_id as usize) < vocab_size, "Sampled token out of range");
            counter[i * vocab_size + sampled_id as usize] += 1;
        }
    }

    for i in 0..batch_size {
        for j in 0..vocab_size {
            if counter[i * vocab_size + j] > 0 {
                assert!(
                    high_prob_token_sets[i].contains(&j),
                    "high_prob_token_sets[{}] does not contain {} (appeared {} times)",
                    i,
                    j,
                    counter[i * vocab_size + j]
                );
            }
        }
    }

    println!(
        "batch_size: {}, p: {:.1}, vocab_size: {}, accuracy test passed (backend: {}).",
        batch_size,
        p,
        vocab_size,
        std::any::type_name::<B>()
    );
}

#[uzu_test]
fn test_topp_sampling_match_small() {
    for_each_non_cpu_backend!(|B| {
        test_topp_sampling_from_prob_exact_match_internal::<B>(8, 10, 1024);
    });
}

#[uzu_test]
fn test_topp_sampling_match_large() {
    for_each_non_cpu_backend!(|B| {
        test_topp_sampling_from_prob_exact_match_internal::<B>(32, 50, 4096);
    });
}

#[uzu_test]
fn test_topp_sampling_statistical_large() {
    for_each_non_cpu_backend!(|B| {
        const BATCH: usize = 32;
        const VOCAB: usize = 4096;
        const NUM_DRAWS: usize = 10_000;
        const TOP_P: f32 = 0.9;
        const TOLERANCE_KL: f32 = 0.05;

        let context = <B as Backend>::Context::new().expect("Failed to create Context");

        let kernel =
            SamplingKernel::<B>::new(&context, DataType::F32, BATCH, VOCAB).expect("Failed to create sampling kernel");

        let mut rng = StdRng::seed_from_u64(42);
        let mut logits = vec![0.0f32; BATCH * VOCAB];
        for x in logits.iter_mut() {
            *x = rng.random_range(-6.0f32..6.0f32);
        }

        // Build Top-p renormalised target distribution per row
        let mut probs = vec![0.0f32; BATCH * VOCAB];
        for b in 0..BATCH {
            let row_logits = &logits[b * VOCAB..(b + 1) * VOCAB];
            let dist = cpu_reference_top_p(row_logits, TOP_P);
            probs[b * VOCAB..(b + 1) * VOCAB].copy_from_slice(&dist);
        }

        let output_array = context.create_array_uninitialized(&[BATCH], DataType::U32, "");
        let mut counters = vec![0u32; BATCH * VOCAB];

        for draw in 0..NUM_DRAWS {
            let logits_array = context.create_array_from(&[BATCH * VOCAB], &logits, "");
            let seeds: Vec<u64> = vec![TEST_SAMPLING_SEED + draw as u64; BATCH];
            let seeds_array = context.create_array_from(&[BATCH], &seeds, "");

            let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

            kernel
                .encode(
                    logits_array.buffer().borrow_mut().deref_mut(),
                    seeds_array.buffer().borrow().deref(),
                    0,
                    None,
                    0,
                    output_array.buffer().borrow_mut().deref_mut(),
                    SamplingMethod::Stochastic {
                        temperature: None,
                        top_k: None,
                        top_p: Some(TOP_P),
                        min_p: None,
                        processing_order: SamplingProcessingOrder::TemperatureThenFilters,
                    },
                    BATCH,
                    VOCAB,
                    &mut encoder,
                )
                .expect("encode");
            encoder.end_encoding().submit().wait_until_completed().unwrap();

            let sample_ids: &[u32] = output_array.as_slice();
            for (b, &tok) in sample_ids.iter().enumerate() {
                counters[b * VOCAB + tok as usize] += 1;
            }
        }

        // Compute KL divergence per row
        for b in 0..BATCH {
            let mut kl = 0.0_f32;
            for j in 0..VOCAB {
                let expected = probs[b * VOCAB + j];
                if expected < 1e-12 {
                    continue;
                }
                let observed = (counters[b * VOCAB + j] as f32) / (NUM_DRAWS as f32);
                if observed > 0.0 {
                    kl += observed * (observed.ln() - expected.ln());
                }
            }
            assert!(kl < TOLERANCE_KL, "Row {} KL {:.4} exceeded tolerance {:.3}", b, kl, TOLERANCE_KL);
        }

        println!(
            "Large statistical Top-p test passed (KL < {:.3}, backend: {})",
            TOLERANCE_KL,
            std::any::type_name::<B>()
        );
    });
}

#[uzu_test]
fn perf_topp_128k_vocab() {
    for_each_non_cpu_backend!(|B| {
        const BATCH: usize = 8;
        const VOCAB: usize = 128000;
        const TOP_P: f32 = 0.9;

        let context = <B as Backend>::Context::new().expect("Failed to create Context");

        let kernel =
            SamplingKernel::<B>::new(&context, DataType::F32, BATCH, VOCAB).expect("Failed to create sampling kernel");

        let mut rng = StdRng::seed_from_u64(123);
        let mut logits = vec![0.0f32; BATCH * VOCAB];
        for x in logits.iter_mut() {
            *x = rng.random_range(-6.0f32..6.0f32);
        }

        let logits_array = context.create_array_from(&[BATCH * VOCAB], &logits, "");
        let seeds: Vec<u64> = vec![TEST_SAMPLING_SEED; BATCH];
        let seeds_array = context.create_array_from(&[BATCH], &seeds, "");
        let output_array = context.create_array_uninitialized(&[BATCH], DataType::U32, "");

        let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

        kernel
            .encode(
                logits_array.buffer().borrow_mut().deref_mut(),
                seeds_array.buffer().borrow().deref(),
                0,
                None,
                0,
                output_array.buffer().borrow_mut().deref_mut(),
                SamplingMethod::Stochastic {
                    temperature: None,
                    top_k: None,
                    top_p: Some(TOP_P),
                    min_p: None,
                    processing_order: SamplingProcessingOrder::TemperatureThenFilters,
                },
                BATCH,
                VOCAB,
                &mut encoder,
            )
            .expect("encode");

        let host_timer = std::time::Instant::now();
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        let host_elapsed_ms = host_timer.elapsed().as_secs_f64() * 1e3;

        let gpu_time_ms = completed.gpu_execution_time().as_secs_f64() * 1e3;
        println!(
            "Top-p sampling perf (batch={}, vocab={}, backend={}): GPU={:.2} ms, Host-side={:.2} ms",
            BATCH,
            VOCAB,
            std::any::type_name::<B>(),
            gpu_time_ms,
            host_elapsed_ms
        );

        let sample_ids: &[u32] = output_array.as_slice();
        for &tok in sample_ids {
            assert!((tok as usize) < VOCAB, "Sampled id out of range");
        }
    });
}
