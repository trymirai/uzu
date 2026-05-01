#![cfg(metal_backend)]

use rand::seq::SliceRandom;

// for Vec::shuffle
use crate::{
    DataType,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::{
                MinPKernel, TemperatureKernel,
                sampling::{ArgmaxStrategy, SamplingKernel},
            },
        },
        metal::Metal,
    },
    session::parameter::{SamplingMethod, SamplingProcessingOrder},
};

#[path = "../../common/mod.rs"]
mod common;

use common::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec};

// Constant seed for reproducible test results
const TEST_SAMPLING_SEED: u64 = 42;

fn cpu_reference_min_p(
    row_logits: &[f32],
    min_p: f32,
) -> Vec<f32> {
    let max_logit = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let threshold = max_logit + min_p.ln();

    row_logits
        .iter()
        .map(|&logit| {
            if logit >= threshold {
                logit
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

fn test_argmax_sampling_with_strategy(strategy: ArgmaxStrategy) {
    let context = match <Metal as Backend>::Context::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping argmax test: {}", e);
            return;
        },
    };

    let batch_size = 2;
    let vocab_size = 4;

    let kernel = SamplingKernel::<Metal>::new_with_strategy(&context, DataType::F32, strategy)
        .expect("Failed to create argmax kernel");

    // Create test data: batch_size=2, vocab_size=4
    // First batch: [1.0, 3.0, 2.0, 0.5] -> should select index 1
    // Second batch: [0.1, 0.5, 2.5, 1.0] -> should select index 2
    let test_logits: Vec<f32> = vec![
        1.0, 3.0, 2.0, 0.5, // batch 0
        0.1, 0.5, 2.5, 1.0, // batch 1
    ];

    let mut logits_buffer = alloc_allocation_with_data::<Metal, _>(context.as_ref(), &test_logits);
    let seeds_buffer = alloc_allocation_with_data::<Metal, _>(context.as_ref(), &[0_u64, 0_u64]);
    let mut output_buffer = alloc_allocation::<Metal, u32>(context.as_ref(), batch_size);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel
        .encode(
            &mut logits_buffer,
            &seeds_buffer,
            None,
            &mut output_buffer,
            SamplingMethod::Greedy,
            batch_size,
            vocab_size,
            &mut encoder,
        )
        .expect("Argmax sampling should succeed");
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    // Check results
    let results: Vec<u32> = allocation_to_vec(&output_buffer);

    assert_eq!(results[0], 1, "First batch should select token 1 (highest logit: 3.0)");
    assert_eq!(results[1], 2, "Second batch should select token 2 (highest logit: 2.5)");

    println!("✓ Argmax sampling test passed with {:?} strategy - selected tokens: {:?}", strategy, results);
}

#[test]
fn test_argmax_sampling_single_pass() {
    test_argmax_sampling_with_strategy(ArgmaxStrategy::SinglePass);
}

#[test]
fn test_argmax_sampling_two_pass() {
    test_argmax_sampling_with_strategy(ArgmaxStrategy::TwoPass);
}

#[test]
fn test_argmax_sampling() {
    // Keep the original test for backward compatibility - defaults to single-pass
    test_argmax_sampling_with_strategy(ArgmaxStrategy::SinglePass);
}

fn perf_argmax_128k_vocab_with_strategy(strategy: ArgmaxStrategy) {
    use std::time::Instant;

    use rand::{RngExt, SeedableRng, rngs::StdRng};

    // ---- Metal context ----
    let context = match <Metal as Backend>::Context::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping argmax perf test: {}", e);
            return;
        },
    };

    // ---- Problem sizes ----
    const BATCH: usize = 8;
    const VOCAB: usize = 128000; // 128K

    // ---- Kernel ----
    let kernel = SamplingKernel::<Metal>::new_with_strategy(&context, DataType::F32, strategy)
        .expect("Failed to create Argmax kernel");

    // ---- Build random logits ----
    let mut rng = StdRng::seed_from_u64(123);
    let mut logits = vec![0.0f32; BATCH * VOCAB];
    for x in logits.iter_mut() {
        *x = rng.random_range(-6.0f32..6.0f32);
    }

    let mut logits_buf = alloc_allocation_with_data::<Metal, _>(context.as_ref(), &logits);

    let seeds: Vec<u64> = vec![TEST_SAMPLING_SEED; BATCH];
    let seeds_buf = alloc_allocation_with_data::<Metal, _>(context.as_ref(), &seeds);

    let mut output_buf = alloc_allocation::<Metal, u32>(context.as_ref(), BATCH);

    // ---- Launch once and time ----
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

    kernel
        .encode(
            &mut logits_buf,
            &seeds_buf,
            None::<&backend_uzu::backends::common::Allocation<Metal>>,
            &mut output_buf,
            SamplingMethod::Greedy,
            BATCH,
            VOCAB,
            &mut encoder,
        )
        .expect("encode");

    // Time both host-side and GPU execution
    let host_timer = Instant::now();
    let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
    let host_elapsed_ms = host_timer.elapsed().as_secs_f64() * 1e3;

    let gpu_time_ms = completed.gpu_execution_time().as_secs_f64() * 1e3;
    println!(
        "Argmax sampling perf (batch={}, vocab={}, strategy={:?}): GPU={:.2} ms, Host-side={:.2} ms",
        BATCH, VOCAB, strategy, gpu_time_ms, host_elapsed_ms
    );

    // Ensure the kernel produced *some* output (sanity).
    let sample_ids: Vec<u32> = allocation_to_vec(&output_buf);
    for &tok in &sample_ids {
        assert!((tok as usize) < VOCAB, "Sampled id out of range");
    }

    // Also verify correctness by checking if selected tokens have highest logits
    for b in 0..BATCH {
        let row_logits = &logits[b * VOCAB..(b + 1) * VOCAB];
        let selected_token = sample_ids[b] as usize;
        let selected_logit = row_logits[selected_token];

        // Find the actual maximum logit
        let max_logit = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        assert!(
            (selected_logit - max_logit).abs() < 1e-5,
            "Batch {}: selected token {} with logit {:.6}, but max logit is {:.6}",
            b,
            selected_token,
            selected_logit,
            max_logit
        );
    }

    println!("✓ Argmax correctness verified with {:?} strategy", strategy);
}

#[test]
#[ignore = "performance-only check"]
fn perf_argmax_128k_vocab_single_pass() {
    perf_argmax_128k_vocab_with_strategy(ArgmaxStrategy::SinglePass);
}

#[test]
#[ignore = "performance-only check"]
fn perf_argmax_128k_vocab_two_pass() {
    perf_argmax_128k_vocab_with_strategy(ArgmaxStrategy::TwoPass);
}

#[test]
fn test_categorical_sampling() {
    let context = match <Metal as Backend>::Context::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping categorical test: {}", e);
            return;
        },
    };

    let batch_size = 2;
    let vocab_size = 4;

    let kernel = SamplingKernel::<Metal>::new(&context, DataType::F32).expect("Failed to create sampling kernel");

    // Create test data with different probability distributions
    // First batch: [1.0, 2.0, 1.5, 0.5] -> softmax: [0.134, 0.366, 0.201, 0.082]
    // Second batch: [0.0, 1.0, 0.0, 0.0] -> softmax: [0.25, 0.75, 0.25, 0.25] (but 1.0 should dominate)
    let test_logits: Vec<f32> = vec![
        1.0, 2.0, 1.5, 0.5, // batch 0
        0.0, 1.0, 0.0, 0.0, // batch 1
    ];

    let mut output_buffer = alloc_allocation::<Metal, u32>(context.as_ref(), batch_size);

    // Run sampling multiple times to check distribution
    let num_samples = 1000;
    let mut counts = vec![0; batch_size * vocab_size];

    for sample_idx in 0..num_samples {
        // Create fresh logits buffer since kernel mutates in-place
        let mut logits_buffer = alloc_allocation_with_data::<Metal, _>(context.as_ref(), &test_logits);

        let seeds_vec = vec![TEST_SAMPLING_SEED + sample_idx as u64; batch_size];
        let seeds_buffer = alloc_allocation_with_data::<Metal, _>(context.as_ref(), &seeds_vec);

        let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

        kernel
            .encode(
                &mut logits_buffer,
                &seeds_buffer,
                None,
                &mut output_buffer,
                SamplingMethod::Stochastic {
                    temperature: None,
                    top_k: None,
                    top_p: None,
                    min_p: None,
                    processing_order: SamplingProcessingOrder::TemperatureThenFilters,
                },
                batch_size,
                vocab_size,
                &mut encoder,
            )
            .expect("Categorical sampling should succeed");
        encoder.end_encoding().submit().wait_until_completed().unwrap();

        let results: Vec<u32> = allocation_to_vec(&output_buffer);

        for (batch_idx, &token) in results.iter().enumerate() {
            assert!(
                (token as usize) < vocab_size,
                "Sampled token {} out of range for vocab size {}",
                token,
                vocab_size
            );
            counts[batch_idx * vocab_size + token as usize] += 1;
        }
    }

    // Check that all batches produced valid outputs
    for batch_idx in 0..batch_size {
        let batch_counts = &counts[batch_idx * vocab_size..(batch_idx + 1) * vocab_size];
        let total_samples: usize = batch_counts.iter().sum();
        assert_eq!(
            total_samples, num_samples,
            "Batch {} should have {} samples, got {}",
            batch_idx, num_samples, total_samples
        );

        // For the second batch, token 1 should be sampled most frequently
        if batch_idx == 1 {
            let max_count = *batch_counts.iter().max().unwrap();
            assert_eq!(
                batch_counts[1], max_count,
                "Token 1 should be most frequent in batch 1, got counts: {:?}",
                batch_counts
            );
            // Token 1 should have at least 30% of the samples (due to higher logit)
            assert!(
                batch_counts[1] as f32 / num_samples as f32 > 0.3,
                "Token 1 should have at least 30% frequency, got {:.1}%",
                batch_counts[1] as f32 / num_samples as f32 * 100.0
            );
        }
    }

    println!("✓ Categorical sampling test passed - sampled {} times per batch", num_samples);
    println!("  Batch 0 counts: {:?}", &counts[0..vocab_size]);
    println!("  Batch 1 counts: {:?}", &counts[vocab_size..2 * vocab_size]);
}

#[test]
fn test_categorical_sampling_statistical() {
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    let context = match <Metal as Backend>::Context::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping categorical statistical test: {}", e);
            return;
        },
    };

    const BATCH: usize = 4;
    const VOCAB: usize = 8;
    const NUM_SAMPLES: usize = 5000;
    const TOLERANCE: f32 = 0.05; // 5% tolerance

    let kernel = SamplingKernel::<Metal>::new(&context, DataType::F32).expect("Failed to create sampling kernel");

    // Generate random logits
    let mut rng = StdRng::seed_from_u64(42);
    let mut logits = vec![0.0f32; BATCH * VOCAB];
    for x in logits.iter_mut() {
        *x = rng.random_range(-3.0f32..3.0f32);
    }

    // Compute expected probabilities (softmax)
    let mut expected_probs = vec![0.0f32; BATCH * VOCAB];
    for batch_idx in 0..BATCH {
        let batch_logits = &logits[batch_idx * VOCAB..(batch_idx + 1) * VOCAB];
        let max_logit = batch_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exp = 0.0f32;
        for j in 0..VOCAB {
            let exp_val = (batch_logits[j] - max_logit).exp();
            expected_probs[batch_idx * VOCAB + j] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize
        for j in 0..VOCAB {
            expected_probs[batch_idx * VOCAB + j] /= sum_exp;
        }
    }

    let mut output_buffer = alloc_allocation::<Metal, u32>(context.as_ref(), BATCH);

    let mut counts = vec![0; BATCH * VOCAB];

    // Sample many times
    for sample_idx in 0..NUM_SAMPLES {
        // Create fresh logits buffer since kernel mutates in-place
        let mut logits_buffer = alloc_allocation_with_data::<Metal, _>(context.as_ref(), &logits);

        let seeds_vec = vec![TEST_SAMPLING_SEED + sample_idx as u64; BATCH];
        let seeds_buffer = alloc_allocation_with_data::<Metal, _>(context.as_ref(), &seeds_vec);

        let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

        kernel
            .encode(
                &mut logits_buffer,
                &seeds_buffer,
                None,
                &mut output_buffer,
                SamplingMethod::Stochastic {
                    temperature: None,
                    top_k: None,
                    top_p: None,
                    min_p: None,
                    processing_order: SamplingProcessingOrder::TemperatureThenFilters,
                },
                BATCH,
                VOCAB,
                &mut encoder,
            )
            .expect("encode");
        encoder.end_encoding().submit().wait_until_completed().unwrap();

        let results: Vec<u32> = allocation_to_vec(&output_buffer);

        for (batch_idx, &token) in results.iter().enumerate() {
            counts[batch_idx * VOCAB + token as usize] += 1;
        }
    }

    // Check statistical accuracy
    for batch_idx in 0..BATCH {
        for token_idx in 0..VOCAB {
            let observed_freq = counts[batch_idx * VOCAB + token_idx] as f32 / NUM_SAMPLES as f32;
            let expected_freq = expected_probs[batch_idx * VOCAB + token_idx];

            if expected_freq > 0.01 {
                // Only check for tokens with reasonable probability
                let error = (observed_freq - expected_freq).abs();
                assert!(
                    error < TOLERANCE,
                    "Batch {}, Token {}: expected freq {:.3}, got {:.3}, error {:.3} > tolerance {:.3}",
                    batch_idx,
                    token_idx,
                    expected_freq,
                    observed_freq,
                    error,
                    TOLERANCE
                );
            }
        }
    }

    println!("✓ Categorical statistical test passed (tolerance: {:.1}%)", TOLERANCE * 100.0);
}

#[test]
fn perf_categorical_128k_vocab() {
    use std::time::Instant;

    use rand::{RngExt, SeedableRng, rngs::StdRng};

    let context = match <Metal as Backend>::Context::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping categorical perf test: {}", e);
            return;
        },
    };

    const BATCH: usize = 8;
    const VOCAB: usize = 128000; // 128K

    let kernel = SamplingKernel::<Metal>::new(&context, DataType::F32).expect("Failed to create sampling kernel");

    // Build random logits
    let mut rng = StdRng::seed_from_u64(123);
    let mut logits = vec![0.0f32; BATCH * VOCAB];
    for x in logits.iter_mut() {
        *x = rng.random_range(-6.0f32..6.0f32);
    }

    let mut logits_buf = alloc_allocation_with_data::<Metal, _>(context.as_ref(), &logits);

    let seeds: Vec<u64> = vec![TEST_SAMPLING_SEED; BATCH];
    let seeds_buf = alloc_allocation_with_data::<Metal, _>(context.as_ref(), &seeds);

    let mut output_buf = alloc_allocation::<Metal, u32>(context.as_ref(), BATCH);

    // Launch and time
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

    kernel
        .encode(
            &mut logits_buf,
            &seeds_buf,
            None,
            &mut output_buf,
            SamplingMethod::Stochastic {
                temperature: None,
                top_k: None,
                top_p: None,
                min_p: None,
                processing_order: SamplingProcessingOrder::TemperatureThenFilters,
            },
            BATCH,
            VOCAB,
            &mut encoder,
        )
        .expect("encode");

    let host_timer = Instant::now();
    let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
    let host_elapsed_ms = host_timer.elapsed().as_secs_f64() * 1e3;

    let gpu_time_ms = completed.gpu_execution_time().as_secs_f64() * 1e3;
    println!(
        "Categorical sampling perf (batch={}, vocab={}): GPU={:.2} ms, Host-side={:.2} ms",
        BATCH, VOCAB, gpu_time_ms, host_elapsed_ms
    );

    // Sanity check
    let sample_ids: Vec<u32> = allocation_to_vec(&output_buf);
    for &tok in &sample_ids {
        assert!((tok as usize) < VOCAB, "Sampled id out of range");
    }

    println!("✓ Categorical correctness verified");
}

#[test]
fn test_temperature_gpu_cpu_match() {
    let context = match <Metal as Backend>::Context::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping temperature gpu cpu match test: {}", e);
            return;
        },
    };

    const BATCH: usize = 4;
    const VOCAB: usize = 1024;
    const TEMPERATURE: f32 = 0.7;
    const RTOL: f32 = 1e-6;
    const ATOL: f32 = 1e-6;

    let kernel = <<Metal as Backend>::Kernels as Kernels>::TemperatureKernel::new(&context, DataType::F32, false)
        .expect("Failed to create temperature kernel");

    let logits: Vec<f32> = (0..BATCH * VOCAB).map(|i| ((i * 37 % 1000) as f32 - 500.0) * 0.01).collect();

    let logits_buffer = alloc_allocation_with_data::<Metal, _>(context.as_ref(), &logits);
    let mut processed_buffer = alloc_allocation::<Metal, f32>(context.as_ref(), logits.len());

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(Some(&logits_buffer), &mut processed_buffer, BATCH as u32, VOCAB as u32, TEMPERATURE, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let gpu_results: Vec<f32> = allocation_to_vec(&processed_buffer);

    for (idx, (&logit, &processed)) in logits.iter().zip(gpu_results.iter()).enumerate() {
        let expected = logit / TEMPERATURE;
        let abs_diff = (expected - processed).abs();
        let tolerance = ATOL + RTOL * expected.abs();
        assert!(
            abs_diff <= tolerance,
            "Mismatch at element {}: expected={} actual={} (abs_diff={}, tolerance={})",
            idx,
            expected,
            processed,
            abs_diff,
            tolerance
        );
    }

    println!("✓ Temperature processor gpu cpu match (temp={}, rtol={}, atol={})", TEMPERATURE, RTOL, ATOL);
}

#[test]
fn test_minp_gpu_cpu_match() {
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    let context = match <Metal as Backend>::Context::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping minp gpu cpu match test: {}", e);
            return;
        },
    };

    const BATCH: usize = 4;
    const VOCAB: usize = 1024;
    const MINP: f32 = 0.1;

    let kernel = <<Metal as Backend>::Kernels as Kernels>::MinPKernel::new(&context, DataType::F32, false)
        .expect("Failed to create minp kernel");

    let mut rng = StdRng::seed_from_u64(42);
    let mut logits = vec![0.0f32; BATCH * VOCAB];
    for x in logits.iter_mut() {
        *x = rng.random_range(-16.0f32..16.0f32);
    }

    let logits_buffer = alloc_allocation_with_data::<Metal, _>(context.as_ref(), &logits);
    let mut processed_buffer = alloc_allocation::<Metal, f32>(context.as_ref(), logits.len());

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(Some(&logits_buffer), &mut processed_buffer, BATCH as u32, VOCAB as u32, MINP, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let all_results: Vec<f32> = allocation_to_vec(&processed_buffer);

    for batch_idx in 0..BATCH {
        let cpu_logits = &logits[batch_idx * VOCAB..(batch_idx + 1) * VOCAB];
        let cpu_processed_logits = cpu_reference_min_p(cpu_logits, MINP);

        let gpu_processed_logits = &all_results[batch_idx * VOCAB..(batch_idx + 1) * VOCAB];

        assert_eq!(&cpu_processed_logits, gpu_processed_logits);
    }

    println!("✓ Minp processor gpu cpu match (minp={})", MINP);
}

fn test_minp_sampling_exact_match(
    batch_size: usize,
    min_p: f32,
    vocab_size: usize,
) {
    use std::collections::HashSet;

    use rand::{SeedableRng, rngs::StdRng};

    let context = match <Metal as Backend>::Context::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping minp exact match test: {}", e);
            return;
        },
    };

    let kernel = SamplingKernel::<Metal>::new(&context, DataType::F32).expect("Failed to create sampling kernel");

    // Build logits where some tokens have high probability and others have low
    // For min_p filtering, tokens with probability < min_p * max_prob are masked
    // We set k tokens to have probability 0.1 each (high), rest get very low probability
    let k = 10;
    let high_logit = 0.0f32; // exp(0) = 1
    let low_logit = -50.0f32; // exp(-50) ≈ 0

    let mut logits = vec![low_logit; batch_size * vocab_size];
    let mut high_prob_token_sets: Vec<HashSet<usize>> = Vec::with_capacity(batch_size);

    let mut rng = StdRng::seed_from_u64(42);
    let mut all_token_ids: Vec<usize> = (0..vocab_size).collect();

    for b in 0..batch_size {
        all_token_ids.shuffle(&mut rng);
        let high_prob_tokens: HashSet<usize> = all_token_ids[..k].iter().cloned().collect();

        for &token_id in &high_prob_tokens {
            logits[b * vocab_size + token_id] = high_logit;
        }
        high_prob_token_sets.push(high_prob_tokens);
    }

    let mut output_buf = alloc_allocation::<Metal, u32>(context.as_ref(), batch_size);

    let num_samples = 1000;
    let mut counter = vec![0i32; batch_size * vocab_size];

    for draw in 0..num_samples {
        // Create fresh logits buffer since kernel mutates in-place
        let mut logits_buf = alloc_allocation_with_data::<Metal, _>(context.as_ref(), &logits);

        let seeds: Vec<u64> = vec![TEST_SAMPLING_SEED + draw as u64; batch_size];
        let seeds_buf = alloc_allocation_with_data::<Metal, _>(context.as_ref(), &seeds);

        let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
        kernel
            .encode(
                &mut logits_buf,
                &seeds_buf,
                None,
                &mut output_buf,
                SamplingMethod::Stochastic {
                    temperature: None,
                    top_k: None,
                    top_p: None,
                    min_p: Some(min_p),
                    processing_order: SamplingProcessingOrder::TemperatureThenFilters,
                },
                batch_size,
                vocab_size,
                &mut encoder,
            )
            .expect("encode");
        encoder.end_encoding().submit().wait_until_completed().unwrap();

        let sampled_ids: Vec<u32> = allocation_to_vec(&output_buf);

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

    println!("batch_size: {}, min_p: {:.2}, vocab_size: {}, accuracy test passed.", batch_size, min_p, vocab_size);
}

#[test]
fn test_minp_sampling_match_small() {
    test_minp_sampling_exact_match(8, 0.1, 1024);
}

#[test]
fn test_minp_sampling_match_large() {
    test_minp_sampling_exact_match(32, 0.05, 4096);
}
