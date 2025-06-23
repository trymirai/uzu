#[cfg(test)]
use metal::{Device, MTLResourceOptions};
use rand::seq::SliceRandom; // for Vec::shuffle
use uzu::{
    backends::metal::{
        KernelDataType, MTLContext,
        kernel::{
            SamplingKernel,
            sampling::{ArgmaxStrategy, CategoricalStrategy},
        },
        metal_extensions::command_buffer_extensions::CommandBufferTimingAccess,
    },
    session::sampling_config::SamplingConfig,
};

// Constant seed for reproducible test results
const TEST_SAMPLING_SEED: u64 = 42;

fn create_test_context() -> Result<MTLContext, String> {
    let device = Device::system_default().ok_or("No Metal device available")?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue)
        .map_err(|e| format!("Failed to create MTLContext: {:?}", e))
}

fn cpu_reference_top_p(
    row_logits: &[f32],
    top_p: f32,
) -> Vec<f32> {
    // 1. soft‑max (unnormalised)
    let max_logit =
        row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut softmax: Vec<(usize, f32)> = row_logits
        .iter()
        .enumerate()
        .map(|(idx, &z)| (idx, (z - max_logit).exp()))
        .collect();

    // 2. sort descending
    softmax.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 3. accumulate until cumulative ≥ top_p
    let mut cum = 0.0_f32;
    let mut cutoff = 0;
    while cutoff < softmax.len() && cum < top_p {
        cum += softmax[cutoff].1;
        cutoff += 1;
    }

    // 4. renormalise the kept mass
    let mut dist = vec![0.0f32; row_logits.len()];
    if cum > 0.0 {
        let renorm = 1.0 / cum;
        for (idx, p) in &softmax[..cutoff] {
            dist[*idx] = *p * renorm;
        }
    }
    dist
}

fn test_argmax_sampling_with_strategy(strategy: ArgmaxStrategy) {
    let context = match create_test_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping argmax test: {}", e);
            return;
        },
    };

    let batch_size = 2;
    let vocab_size = 4;

    let kernel = SamplingKernel::new_with_strategy(
        &context,
        KernelDataType::Float32,
        batch_size,
        vocab_size,
        strategy,
        TEST_SAMPLING_SEED,
    )
    .expect("Failed to create argmax kernel");

    // Create test data: batch_size=2, vocab_size=4
    // First batch: [1.0, 3.0, 2.0, 0.5] -> should select index 1
    // Second batch: [0.1, 0.5, 2.5, 1.0] -> should select index 2
    let test_logits: Vec<f32> = vec![
        1.0, 3.0, 2.0, 0.5, // batch 0
        0.1, 0.5, 2.5, 1.0, // batch 1
    ];

    let logits_buffer = context.device.new_buffer_with_data(
        test_logits.as_ptr() as *const _,
        (test_logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = context.device.new_buffer(
        (batch_size * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer_ref = context.command_queue.new_command_buffer();
    let command_buffer = command_buffer_ref.to_owned();

    // Run sampling
    kernel
        .encode(
            &SamplingConfig::argmax(),
            &logits_buffer,
            &output_buffer,
            batch_size,
            vocab_size,
            &command_buffer,
        )
        .expect("Argmax sampling should succeed");

    command_buffer_ref.commit();
    command_buffer_ref.wait_until_completed();

    // Check results
    let result_ptr = output_buffer.contents() as *const u32;
    let results = unsafe { std::slice::from_raw_parts(result_ptr, batch_size) };

    assert_eq!(
        results[0], 1,
        "First batch should select token 1 (highest logit: 3.0)"
    );
    assert_eq!(
        results[1], 2,
        "Second batch should select token 2 (highest logit: 2.5)"
    );

    println!(
        "✓ Argmax sampling test passed with {:?} strategy - selected tokens: {:?}",
        strategy, results
    );
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

fn test_topp_sampling_from_prob_exact_match(
    batch_size: usize,
    k: usize,
    vocab_size: usize,
) {
    use std::collections::HashSet;

    use rand::{SeedableRng, rngs::StdRng};

    let context = match create_test_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping topP exact match test: {}", e);
            return;
        },
    };

    // Calculate top_p threshold
    let p = (k as f32) * 0.1;

    // Create kernel
    let kernel = SamplingKernel::new(
        &context,
        KernelDataType::Float32,
        batch_size,
        vocab_size,
        TEST_SAMPLING_SEED,
    )
    .expect("Failed to create sampling kernel");

    // Build probability table
    let low_prob = (1.0 - p) / (vocab_size - k) as f32;
    let mut probs = vec![low_prob; batch_size * vocab_size];
    let mut high_prob_token_sets: Vec<HashSet<usize>> =
        Vec::with_capacity(batch_size);

    // Use std::sample equivalent (Fisher-Yates shuffle + take first k)
    let mut rng = StdRng::seed_from_u64(42); // Deterministic for reproducibility
    let mut all_token_ids: Vec<usize> = (0..vocab_size).collect();

    for b in 0..batch_size {
        // Equivalent to std::sample - shuffle and take first k
        all_token_ids.shuffle(&mut rng);
        let high_prob_tokens: HashSet<usize> =
            all_token_ids[..k].iter().cloned().collect();

        // Set selected tokens to high probability
        for &token_id in &high_prob_tokens {
            probs[b * vocab_size + token_id] = 0.1;
        }
        high_prob_token_sets.push(high_prob_tokens);
    }

    // Convert to logits more carefully to avoid -∞
    let logits: Vec<f32> = probs
        .iter()
        .map(|&prob| {
            if prob <= 0.0 {
                -50.0 // Very negative but finite value instead of -∞
            } else {
                prob.ln()
            }
        })
        .collect();

    // GPU buffers
    let logits_buf = context.device.new_buffer_with_data(
        logits.as_ptr() as *const _,
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_buf = context.device.new_buffer(
        (batch_size * std::mem::size_of::<u32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Counter for samples
    let num_samples = 1000;
    let mut counter = vec![0i32; batch_size * vocab_size];

    // Draw samples
    for _draw in 0..num_samples {
        let cb_ref = context.command_queue.new_command_buffer();
        let cb = cb_ref.to_owned();
        kernel
            .encode(
                &SamplingConfig::top_p(p),
                &logits_buf,
                &output_buf,
                batch_size,
                vocab_size,
                &cb,
            )
            .expect("encode");
        cb_ref.commit();
        cb_ref.wait_until_completed();

        let ptr = output_buf.contents() as *const u32;
        let sampled_ids =
            unsafe { std::slice::from_raw_parts(ptr, batch_size) };

        for (i, &sampled_id) in sampled_ids.iter().enumerate() {
            assert!(
                (sampled_id as usize) < vocab_size,
                "Sampled token out of range"
            );
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
        "batch_size: {}, p: {:.1}, vocab_size: {}, accuracy test passed.",
        batch_size, p, vocab_size
    );
}

#[test]
fn test_topp_sampling_match_small() {
    test_topp_sampling_from_prob_exact_match(8, 10, 1024);
}

#[test]
fn test_topp_sampling_match_large() {
    test_topp_sampling_from_prob_exact_match(32, 50, 4096);
}

#[test]
fn test_topp_sampling_statistical_large() {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    // ===== 1. Create Metal context =====
    let context = match create_test_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping large statistical test: {}", e);
            return;
        },
    };

    // ===== 2. Problem sizes =====
    const BATCH: usize = 32;
    const VOCAB: usize = 4096;
    const NUM_DRAWS: usize = 10_000;
    const TOP_P: f32 = 0.9;
    const TOLERANCE_KL: f32 = 0.05;

    // ===== 3. Build kernel =====
    let kernel = SamplingKernel::new(
        &context,
        KernelDataType::Float32,
        BATCH,
        VOCAB,
        TEST_SAMPLING_SEED,
    )
    .expect("Failed to create sampling kernel");

    // ===== 4. Generate reproducible random logits =====
    let mut rng = StdRng::seed_from_u64(42);
    let mut logits = vec![0.0f32; BATCH * VOCAB];
    for x in logits.iter_mut() {
        // Uniform[-6, 6]  (broad range avoids numerical under/overflow)
        *x = rng.random_range(-6.0f32..6.0f32);
    }

    // ===== 5. Build Top‑p renormalised target distribution per row =====
    let mut probs = vec![0.0f32; BATCH * VOCAB];
    for b in 0..BATCH {
        let row_logits = &logits[b * VOCAB..(b + 1) * VOCAB];
        let dist = cpu_reference_top_p(row_logits, TOP_P);
        probs[b * VOCAB..(b + 1) * VOCAB].copy_from_slice(&dist);
    }

    // ===== 6. Upload logits to GPU =====
    let logits_buf = context.device.new_buffer_with_data(
        logits.as_ptr() as *const _,
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // ===== 7. Allocate output & counters =====
    let output_buf = context.device.new_buffer(
        (BATCH * std::mem::size_of::<u32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let mut counters = vec![0u32; BATCH * VOCAB];

    // ===== 8. Draw NUM_DRAWS samples =====
    for _ in 0..NUM_DRAWS {
        let cb_ref = context.command_queue.new_command_buffer();
        let cb = cb_ref.to_owned();

        kernel
            .encode(
                &SamplingConfig::top_p(TOP_P),
                &logits_buf,
                &output_buf,
                BATCH,
                VOCAB,
                &cb,
            )
            .expect("encode");

        cb_ref.commit();
        cb_ref.wait_until_completed();

        let ptr = output_buf.contents() as *const u32;
        let sample_ids = unsafe { std::slice::from_raw_parts(ptr, BATCH) };
        for (b, &tok) in sample_ids.iter().enumerate() {
            counters[b * VOCAB + tok as usize] += 1;
        }
    }

    // ===== 9. Compute KL divergence per row =====
    for b in 0..BATCH {
        let mut kl = 0.0_f32;
        for j in 0..VOCAB {
            let expected = probs[b * VOCAB + j];
            if expected < 1e-12 {
                continue;
            } // ignore tiny
            let observed =
                (counters[b * VOCAB + j] as f32) / (NUM_DRAWS as f32);
            if observed > 0.0 {
                kl += observed * (observed.ln() - expected.ln());
            }
        }
        assert!(
            kl < TOLERANCE_KL,
            "Row {} KL {:.4} exceeded tolerance {:.3}",
            b,
            kl,
            TOLERANCE_KL
        );
    }

    println!(
        "✓ Large statistical Top‑p test passed (KL < {:.3})",
        TOLERANCE_KL
    );
}

#[test]
#[ignore]
fn perf_topp_128k_vocab() {
    use std::time::Instant;

    use rand::{Rng, SeedableRng, rngs::StdRng};

    // ---- Metal context ----
    let context = match create_test_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping perf test: {}", e);
            return;
        },
    };

    // ---- Problem sizes ----
    const BATCH: usize = 8;
    const VOCAB: usize = 128000; // 128K
    const TOP_P: f32 = 0.9;

    // ---- Kernel ----
    let kernel = SamplingKernel::new(
        &context,
        KernelDataType::Float32,
        BATCH,
        VOCAB,
        TEST_SAMPLING_SEED,
    )
    .expect("Failed to create sampling kernel");

    // ---- Build random logits ----
    let mut rng = StdRng::seed_from_u64(123);
    let mut logits = vec![0.0f32; BATCH * VOCAB];
    for x in logits.iter_mut() {
        *x = rng.random_range(-6.0f32..6.0f32);
    }

    let logits_buf = context.device.new_buffer_with_data(
        logits.as_ptr() as *const _,
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_buf = context.device.new_buffer(
        (BATCH * std::mem::size_of::<u32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // ---- Launch once and time ----
    let cb_ref = context.command_queue.new_command_buffer();
    let cb = cb_ref.to_owned();

    kernel
        .encode(
            &SamplingConfig::top_p(TOP_P),
            &logits_buf,
            &output_buf,
            BATCH,
            VOCAB,
            &cb,
        )
        .expect("encode");

    // Time both host-side and GPU execution
    let host_timer = Instant::now();
    cb_ref.commit();
    cb_ref.wait_until_completed();
    let host_elapsed_ms = host_timer.elapsed().as_secs_f64() * 1e3;

    // Get actual GPU execution time
    let gpu_elapsed_ms = cb.gpu_execution_time_ms();

    match gpu_elapsed_ms {
        Some(gpu_time) => {
            println!(
                "Top‑p sampling perf (batch={}, vocab={}): GPU={:.2} ms, Host-side={:.2} ms",
                BATCH, VOCAB, gpu_time, host_elapsed_ms
            );
        },
        None => {
            println!(
                "Top‑p sampling perf (batch={}, vocab={}): Host-side={:.2} ms (GPU timing unavailable)",
                BATCH, VOCAB, host_elapsed_ms
            );
        },
    }

    // Ensure the kernel produced *some* output (sanity).
    let ptr = output_buf.contents() as *const u32;
    let sample_ids = unsafe { std::slice::from_raw_parts(ptr, BATCH) };
    for &tok in sample_ids {
        assert!((tok as usize) < VOCAB, "Sampled id out of range");
    }
}

fn perf_argmax_128k_vocab_with_strategy(strategy: ArgmaxStrategy) {
    use std::time::Instant;

    use rand::{Rng, SeedableRng, rngs::StdRng};

    // ---- Metal context ----
    let context = match create_test_context() {
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
    let kernel = SamplingKernel::new_with_strategy(
        &context,
        KernelDataType::Float32,
        BATCH,
        VOCAB,
        strategy,
        TEST_SAMPLING_SEED,
    )
    .expect("Failed to create Argmax kernel");

    // ---- Build random logits ----
    let mut rng = StdRng::seed_from_u64(123);
    let mut logits = vec![0.0f32; BATCH * VOCAB];
    for x in logits.iter_mut() {
        *x = rng.random_range(-6.0f32..6.0f32);
    }

    let logits_buf = context.device.new_buffer_with_data(
        logits.as_ptr() as *const _,
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_buf = context.device.new_buffer(
        (BATCH * std::mem::size_of::<u32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // ---- Launch once and time ----
    let cb_ref = context.command_queue.new_command_buffer();
    let cb = cb_ref.to_owned();

    kernel
        .encode(
            &SamplingConfig::argmax(),
            &logits_buf,
            &output_buf,
            BATCH,
            VOCAB,
            &cb,
        )
        .expect("encode");

    // Time both host-side and GPU execution
    let host_timer = Instant::now();
    cb_ref.commit();
    cb_ref.wait_until_completed();
    let host_elapsed_ms = host_timer.elapsed().as_secs_f64() * 1e3;

    // Get actual GPU execution time
    let gpu_elapsed_ms = cb.gpu_execution_time_ms();

    match gpu_elapsed_ms {
        Some(gpu_time) => {
            println!(
                "Argmax sampling perf (batch={}, vocab={}, strategy={:?}): GPU={:.2} ms, Host-side={:.2} ms",
                BATCH, VOCAB, strategy, gpu_time, host_elapsed_ms
            );
        },
        None => {
            println!(
                "Argmax sampling perf (batch={}, vocab={}, strategy={:?}): Host-side={:.2} ms (GPU timing unavailable)",
                BATCH, VOCAB, strategy, host_elapsed_ms
            );
        },
    }

    // Ensure the kernel produced *some* output (sanity).
    let ptr = output_buf.contents() as *const u32;
    let sample_ids = unsafe { std::slice::from_raw_parts(ptr, BATCH) };
    for &tok in sample_ids {
        assert!((tok as usize) < VOCAB, "Sampled id out of range");
    }

    // Also verify correctness by checking if selected tokens have highest logits
    for b in 0..BATCH {
        let row_logits = &logits[b * VOCAB..(b + 1) * VOCAB];
        let selected_token = sample_ids[b] as usize;
        let selected_logit = row_logits[selected_token];

        // Find the actual maximum logit
        let max_logit =
            row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

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
fn test_categorical_sampling() {
    let context = match create_test_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping categorical test: {}", e);
            return;
        },
    };

    let batch_size = 2;
    let vocab_size = 4;

    let kernel = SamplingKernel::new(
        &context,
        KernelDataType::Float32,
        batch_size,
        vocab_size,
        TEST_SAMPLING_SEED,
    )
    .expect("Failed to create sampling kernel");

    // Create test data with different probability distributions
    // First batch: [1.0, 2.0, 1.5, 0.5] -> softmax: [0.134, 0.366, 0.201, 0.082]
    // Second batch: [0.0, 1.0, 0.0, 0.0] -> softmax: [0.25, 0.75, 0.25, 0.25] (but 1.0 should dominate)
    let test_logits: Vec<f32> = vec![
        1.0, 2.0, 1.5, 0.5, // batch 0
        0.0, 1.0, 0.0, 0.0, // batch 1
    ];

    let logits_buffer = context.device.new_buffer_with_data(
        test_logits.as_ptr() as *const _,
        (test_logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = context.device.new_buffer(
        (batch_size * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Run sampling multiple times to check distribution
    let num_samples = 1000;
    let mut counts = vec![0; batch_size * vocab_size];

    for _ in 0..num_samples {
        let command_buffer_ref = context.command_queue.new_command_buffer();
        let command_buffer = command_buffer_ref.to_owned();

        kernel
            .encode(
                &SamplingConfig::categorical(1.0),
                &logits_buffer,
                &output_buffer,
                batch_size,
                vocab_size,
                &command_buffer,
            )
            .expect("Categorical sampling should succeed");

        command_buffer_ref.commit();
        command_buffer_ref.wait_until_completed();

        let result_ptr = output_buffer.contents() as *const u32;
        let results =
            unsafe { std::slice::from_raw_parts(result_ptr, batch_size) };

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
        let batch_counts =
            &counts[batch_idx * vocab_size..(batch_idx + 1) * vocab_size];
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

    println!(
        "✓ Categorical sampling test passed - sampled {} times per batch",
        num_samples
    );
    println!("  Batch 0 counts: {:?}", &counts[0..vocab_size]);
    println!("  Batch 1 counts: {:?}", &counts[vocab_size..2 * vocab_size]);
}

#[test]
fn test_categorical_sampling_statistical() {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    let context = match create_test_context() {
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

    let kernel = SamplingKernel::new(
        &context,
        KernelDataType::Float32,
        BATCH,
        VOCAB,
        TEST_SAMPLING_SEED,
    )
    .expect("Failed to create sampling kernel");

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
        let max_logit =
            batch_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

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

    let logits_buffer = context.device.new_buffer_with_data(
        logits.as_ptr() as *const _,
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = context.device.new_buffer(
        (BATCH * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let mut counts = vec![0; BATCH * VOCAB];

    // Sample many times
    for _ in 0..NUM_SAMPLES {
        let command_buffer_ref = context.command_queue.new_command_buffer();
        let command_buffer = command_buffer_ref.to_owned();

        kernel
            .encode(
                &SamplingConfig::categorical(1.0),
                &logits_buffer,
                &output_buffer,
                BATCH,
                VOCAB,
                &command_buffer,
            )
            .expect("encode");

        command_buffer_ref.commit();
        command_buffer_ref.wait_until_completed();

        let result_ptr = output_buffer.contents() as *const u32;
        let results = unsafe { std::slice::from_raw_parts(result_ptr, BATCH) };

        for (batch_idx, &token) in results.iter().enumerate() {
            counts[batch_idx * VOCAB + token as usize] += 1;
        }
    }

    // Check statistical accuracy
    for batch_idx in 0..BATCH {
        for token_idx in 0..VOCAB {
            let observed_freq = counts[batch_idx * VOCAB + token_idx] as f32
                / NUM_SAMPLES as f32;
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

    println!(
        "✓ Categorical statistical test passed (tolerance: {:.1}%)",
        TOLERANCE * 100.0
    );
}

fn perf_categorical_128k_vocab() {
    use std::time::Instant;

    use rand::{Rng, SeedableRng, rngs::StdRng};

    let context = match create_test_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping categorical perf test: {}", e);
            return;
        },
    };

    const BATCH: usize = 8;
    const VOCAB: usize = 128000; // 128K

    let kernel = SamplingKernel::new(
        &context,
        KernelDataType::Float32,
        BATCH,
        VOCAB,
        TEST_SAMPLING_SEED,
    )
    .expect("Failed to create sampling kernel");

    // Build random logits
    let mut rng = StdRng::seed_from_u64(123);
    let mut logits = vec![0.0f32; BATCH * VOCAB];
    for x in logits.iter_mut() {
        *x = rng.random_range(-6.0f32..6.0f32);
    }

    let logits_buf = context.device.new_buffer_with_data(
        logits.as_ptr() as *const _,
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_buf = context.device.new_buffer(
        (BATCH * std::mem::size_of::<u32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Launch and time
    let cb_ref = context.command_queue.new_command_buffer();
    let cb = cb_ref.to_owned();

    kernel
        .encode(
            &SamplingConfig::categorical(1.0),
            &logits_buf,
            &output_buf,
            BATCH,
            VOCAB,
            &cb,
        )
        .expect("encode");

    let host_timer = Instant::now();
    cb_ref.commit();
    cb_ref.wait_until_completed();
    let host_elapsed_ms = host_timer.elapsed().as_secs_f64() * 1e3;

    let gpu_elapsed_ms = cb.gpu_execution_time_ms();

    match gpu_elapsed_ms {
        Some(gpu_time) => {
            println!(
                "Categorical sampling perf (batch={}, vocab={}): GPU={:.2} ms, Host-side={:.2} ms",
                BATCH, VOCAB, gpu_time, host_elapsed_ms
            );
        },
        None => {
            println!(
                "Categorical sampling perf (batch={}, vocab={}): Host-side={:.2} ms (GPU timing unavailable)",
                BATCH, VOCAB, host_elapsed_ms
            );
        },
    }

    // Sanity check
    let ptr = output_buf.contents() as *const u32;
    let sample_ids = unsafe { std::slice::from_raw_parts(ptr, BATCH) };
    for &tok in sample_ids {
        assert!((tok as usize) < VOCAB, "Sampled id out of range");
    }

    println!("✓ Categorical correctness verified");
}

#[test]
#[ignore]
fn perf_categorical_128k_vocab_test() {
    perf_categorical_128k_vocab();
}

#[test]
#[ignore]
fn perf_argmax_128k_vocab_single_pass() {
    perf_argmax_128k_vocab_with_strategy(ArgmaxStrategy::SinglePass);
}

#[test]
#[ignore]
fn perf_argmax_128k_vocab_two_pass() {
    perf_argmax_128k_vocab_with_strategy(ArgmaxStrategy::TwoPass);
}

fn test_categorical_sampling_with_strategy(strategy: CategoricalStrategy) {
    let context = match create_test_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping categorical test with {:?}: {}", strategy, e);
            return;
        },
    };

    let batch_size = 2;
    let vocab_size = 4;

    let kernel = SamplingKernel::new_with_strategies(
        &context,
        KernelDataType::Float32,
        batch_size,
        vocab_size,
        ArgmaxStrategy::TwoPass,
        strategy,
        TEST_SAMPLING_SEED,
    )
    .expect("Failed to create categorical kernel");

    // Create test data with different probability distributions
    let test_logits: Vec<f32> = vec![
        1.0, 2.0, 1.5, 0.5, // batch 0
        0.0, 1.0, 0.0, 0.0, // batch 1
    ];

    let logits_buffer = context.device.new_buffer_with_data(
        test_logits.as_ptr() as *const _,
        (test_logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = context.device.new_buffer(
        (batch_size * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Run sampling multiple times to check distribution
    let num_samples = 1000;
    let mut counts = vec![0; batch_size * vocab_size];

    for _ in 0..num_samples {
        let command_buffer_ref = context.command_queue.new_command_buffer();
        let command_buffer = command_buffer_ref.to_owned();

        kernel
            .encode(
                &SamplingConfig::categorical(1.0),
                &logits_buffer,
                &output_buffer,
                batch_size,
                vocab_size,
                &command_buffer,
            )
            .expect("Categorical sampling should succeed");

        command_buffer_ref.commit();
        command_buffer_ref.wait_until_completed();

        let result_ptr = output_buffer.contents() as *const u32;
        let results =
            unsafe { std::slice::from_raw_parts(result_ptr, batch_size) };

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
        let batch_counts =
            &counts[batch_idx * vocab_size..(batch_idx + 1) * vocab_size];
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
                "Token 1 should be most frequent in batch 1 with strategy {:?}, got counts: {:?}",
                strategy, batch_counts
            );
        }
    }

    println!(
        "✓ Categorical sampling test passed with {:?} strategy - sampled {} times per batch",
        strategy, num_samples
    );
}

#[test]
fn test_categorical_sampling_single_pass() {
    test_categorical_sampling_with_strategy(CategoricalStrategy::SinglePass);
}

#[test]
fn test_categorical_sampling_two_pass() {
    test_categorical_sampling_with_strategy(CategoricalStrategy::TwoPass);
}

#[test]
fn test_categorical_1pass_vs_2pass_equivalence() {
    let context = match create_test_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping categorical equivalence test: {}", e);
            return;
        },
    };

    let batch_size = 4;
    let vocab_size = 16;
    const NUM_SAMPLES: usize = 5000;
    const TOLERANCE: f32 = 0.05; // 5% tolerance for statistical comparison

    // Create kernels for both strategies
    let kernel_1pass = SamplingKernel::new_with_strategies(
        &context,
        KernelDataType::Float32,
        batch_size,
        vocab_size,
        ArgmaxStrategy::TwoPass,
        CategoricalStrategy::SinglePass,
        TEST_SAMPLING_SEED,
    )
    .expect("Failed to create single-pass categorical kernel");

    let kernel_2pass = SamplingKernel::new_with_strategies(
        &context,
        KernelDataType::Float32,
        batch_size,
        vocab_size,
        ArgmaxStrategy::TwoPass,
        CategoricalStrategy::TwoPass,
        TEST_SAMPLING_SEED,
    )
    .expect("Failed to create two-pass categorical kernel");

    // Create test data with varied probability distributions
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(42);
    let mut test_logits = vec![0.0f32; batch_size * vocab_size];
    for x in test_logits.iter_mut() {
        *x = rng.random_range(-3.0f32..3.0f32);
    }

    let logits_buffer = context.device.new_buffer_with_data(
        test_logits.as_ptr() as *const _,
        (test_logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer_1pass = context.device.new_buffer(
        (batch_size * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer_2pass = context.device.new_buffer(
        (batch_size * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Collect samples from both strategies
    let mut counts_1pass = vec![0; batch_size * vocab_size];
    let mut counts_2pass = vec![0; batch_size * vocab_size];

    for _ in 0..NUM_SAMPLES {
        // Sample from single-pass
        let command_buffer_ref = context.command_queue.new_command_buffer();
        let command_buffer = command_buffer_ref.to_owned();

        kernel_1pass
            .encode(
                &SamplingConfig::categorical(1.0),
                &logits_buffer,
                &output_buffer_1pass,
                batch_size,
                vocab_size,
                &command_buffer,
            )
            .expect("Single-pass encoding should succeed");

        command_buffer_ref.commit();
        command_buffer_ref.wait_until_completed();

        let result_ptr = output_buffer_1pass.contents() as *const u32;
        let results_1pass =
            unsafe { std::slice::from_raw_parts(result_ptr, batch_size) };

        for (batch_idx, &token) in results_1pass.iter().enumerate() {
            counts_1pass[batch_idx * vocab_size + token as usize] += 1;
        }

        // Sample from two-pass
        let command_buffer_ref = context.command_queue.new_command_buffer();
        let command_buffer = command_buffer_ref.to_owned();

        kernel_2pass
            .encode(
                &SamplingConfig::categorical(1.0),
                &logits_buffer,
                &output_buffer_2pass,
                batch_size,
                vocab_size,
                &command_buffer,
            )
            .expect("Two-pass encoding should succeed");

        command_buffer_ref.commit();
        command_buffer_ref.wait_until_completed();

        let result_ptr = output_buffer_2pass.contents() as *const u32;
        let results_2pass =
            unsafe { std::slice::from_raw_parts(result_ptr, batch_size) };

        for (batch_idx, &token) in results_2pass.iter().enumerate() {
            counts_2pass[batch_idx * vocab_size + token as usize] += 1;
        }
    }

    // Compute expected probabilities from logits (reference distribution)
    let mut expected_probs = vec![0.0f32; batch_size * vocab_size];
    for batch_idx in 0..batch_size {
        let batch_logits =
            &test_logits[batch_idx * vocab_size..(batch_idx + 1) * vocab_size];
        let max_logit =
            batch_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exp = 0.0f32;
        for j in 0..vocab_size {
            let exp_val = (batch_logits[j] - max_logit).exp();
            expected_probs[batch_idx * vocab_size + j] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize
        for j in 0..vocab_size {
            expected_probs[batch_idx * vocab_size + j] /= sum_exp;
        }
    }

    // Compare both strategies against expected distribution and each other
    println!(
        "Comparing single-pass vs two-pass categorical sampling distributions:"
    );

    for batch_idx in 0..batch_size {
        println!("Batch {}:", batch_idx);

        // Check that both strategies match the expected distribution
        for token_idx in 0..vocab_size {
            let expected_freq =
                expected_probs[batch_idx * vocab_size + token_idx];

            let observed_freq_1pass =
                counts_1pass[batch_idx * vocab_size + token_idx] as f32
                    / NUM_SAMPLES as f32;
            let observed_freq_2pass =
                counts_2pass[batch_idx * vocab_size + token_idx] as f32
                    / NUM_SAMPLES as f32;

            // Only check tokens with reasonable probability
            if expected_freq > 0.01 {
                let error_1pass = (observed_freq_1pass - expected_freq).abs();
                let error_2pass = (observed_freq_2pass - expected_freq).abs();

                assert!(
                    error_1pass < TOLERANCE,
                    "Single-pass: Batch {}, Token {}: expected freq {:.3}, got {:.3}, error {:.3} > tolerance {:.3}",
                    batch_idx,
                    token_idx,
                    expected_freq,
                    observed_freq_1pass,
                    error_1pass,
                    TOLERANCE
                );

                assert!(
                    error_2pass < TOLERANCE,
                    "Two-pass: Batch {}, Token {}: expected freq {:.3}, got {:.3}, error {:.3} > tolerance {:.3}",
                    batch_idx,
                    token_idx,
                    expected_freq,
                    observed_freq_2pass,
                    error_2pass,
                    TOLERANCE
                );

                // Compare the two strategies directly
                let strategy_diff =
                    (observed_freq_1pass - observed_freq_2pass).abs();
                assert!(
                    strategy_diff < TOLERANCE,
                    "Strategy difference: Batch {}, Token {}: 1-pass {:.3}, 2-pass {:.3}, diff {:.3} > tolerance {:.3}",
                    batch_idx,
                    token_idx,
                    observed_freq_1pass,
                    observed_freq_2pass,
                    strategy_diff,
                    TOLERANCE
                );

                println!(
                    "  Token {}: expected={:.3}, 1-pass={:.3}, 2-pass={:.3}, diff={:.3}",
                    token_idx,
                    expected_freq,
                    observed_freq_1pass,
                    observed_freq_2pass,
                    strategy_diff
                );
            }
        }
    }

    println!(
        "✓ Single-pass and two-pass categorical sampling produce equivalent distributions (tolerance: {:.1}%)",
        TOLERANCE * 100.0
    );
}

fn perf_categorical_128k_vocab_with_strategy(strategy: CategoricalStrategy) {
    use std::time::Instant;

    use rand::{Rng, SeedableRng, rngs::StdRng};

    let context = match create_test_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!(
                "Skipping categorical perf test with {:?}: {}",
                strategy, e
            );
            return;
        },
    };

    const BATCH: usize = 8;
    const VOCAB: usize = 128000; // 128K

    let kernel = SamplingKernel::new_with_strategies(
        &context,
        KernelDataType::Float32,
        BATCH,
        VOCAB,
        ArgmaxStrategy::TwoPass,
        strategy,
        TEST_SAMPLING_SEED,
    )
    .expect("Failed to create sampling kernel");

    // Build random logits
    let mut rng = StdRng::seed_from_u64(123);
    let mut logits = vec![0.0f32; BATCH * VOCAB];
    for x in logits.iter_mut() {
        *x = rng.random_range(-6.0f32..6.0f32);
    }

    let logits_buf = context.device.new_buffer_with_data(
        logits.as_ptr() as *const _,
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_buf = context.device.new_buffer(
        (BATCH * std::mem::size_of::<u32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Launch and time
    let cb_ref = context.command_queue.new_command_buffer();
    let cb = cb_ref.to_owned();

    kernel
        .encode(
            &SamplingConfig::categorical(1.0),
            &logits_buf,
            &output_buf,
            BATCH,
            VOCAB,
            &cb,
        )
        .expect("encode");

    let host_timer = Instant::now();
    cb_ref.commit();
    cb_ref.wait_until_completed();
    let host_elapsed_ms = host_timer.elapsed().as_secs_f64() * 1e3;

    let gpu_elapsed_ms = cb.gpu_execution_time_ms();

    match gpu_elapsed_ms {
        Some(gpu_time) => {
            println!(
                "Categorical sampling perf (batch={}, vocab={}, strategy={:?}): GPU={:.2} ms, Host-side={:.2} ms",
                BATCH, VOCAB, strategy, gpu_time, host_elapsed_ms
            );
        },
        None => {
            println!(
                "Categorical sampling perf (batch={}, vocab={}, strategy={:?}): Host-side={:.2} ms (GPU timing unavailable)",
                BATCH, VOCAB, strategy, host_elapsed_ms
            );
        },
    }

    // Sanity check
    let ptr = output_buf.contents() as *const u32;
    let sample_ids = unsafe { std::slice::from_raw_parts(ptr, BATCH) };
    for &tok in sample_ids {
        assert!((tok as usize) < VOCAB, "Sampled id out of range");
    }

    println!("✓ Categorical correctness verified with {:?} strategy", strategy);
}

#[test]
#[ignore]
fn perf_categorical_128k_vocab_single_pass() {
    perf_categorical_128k_vocab_with_strategy(CategoricalStrategy::SinglePass);
}

#[test]
#[ignore]
fn perf_categorical_128k_vocab_two_pass() {
    perf_categorical_128k_vocab_with_strategy(CategoricalStrategy::TwoPass);
}
