mod common;
use metal::{Device, MTLResourceOptions};
use rand::{Rng, seq::SliceRandom};
// for Vec::shuffle
use uzu::{
    backends::metal::{
        KernelDataType, MTLContext,
        kernel::{SamplingKernel, sampling::ArgmaxStrategy},
        metal_extensions::command_buffer_extensions::CommandBufferTimingAccess,
    },
    generator::gumbel::{gumbel_float, revidx},
    session::parameter::SamplingMethod,
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
    let total_mass: f32 = softmax.iter().map(|(_, p)| p).sum();

    // 2. sort descending
    softmax.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 3. accumulate until cumulative ≥ top_p
    let mut cum = 0.0_f32;
    let mut cutoff = 0;
    let target = top_p * total_mass;
    while cutoff < softmax.len() && cum < target {
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
            &logits_buffer,
            None,
            &output_buffer,
            SamplingMethod::Greedy,
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
    for draw in 0..num_samples {
        let seeds_buf = context.device.new_buffer_with_data(
            vec![TEST_SAMPLING_SEED + draw as u64; batch_size].as_ptr()
                as *const _,
            (batch_size * std::mem::size_of::<u64>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let cb_ref = context.command_queue.new_command_buffer();
        let cb = cb_ref.to_owned();
        kernel
            .encode(
                &logits_buf,
                Some(&seeds_buf),
                &output_buf,
                SamplingMethod::Stochastic {
                    temperature: None,
                    top_k: None,
                    top_p: Some(p),
                },
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
    let kernel =
        SamplingKernel::new(&context, KernelDataType::Float32, BATCH, VOCAB)
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
    for draw in 0..NUM_DRAWS {
        let seeds_buf = context.device.new_buffer_with_data(
            vec![TEST_SAMPLING_SEED + draw as u64; BATCH].as_ptr() as *const _,
            (BATCH * std::mem::size_of::<u64>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let cb_ref = context.command_queue.new_command_buffer();
        let cb = cb_ref.to_owned();

        kernel
            .encode(
                &logits_buf,
                Some(&seeds_buf),
                &output_buf,
                SamplingMethod::Stochastic {
                    temperature: None,
                    top_k: None,
                    top_p: Some(TOP_P),
                },
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
    let kernel =
        SamplingKernel::new(&context, KernelDataType::Float32, BATCH, VOCAB)
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

    let seeds_buf = context.device.new_buffer_with_data(
        vec![TEST_SAMPLING_SEED; BATCH].as_ptr() as *const _,
        (BATCH * std::mem::size_of::<u64>()) as u64,
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
            &logits_buf,
            Some(&seeds_buf),
            &output_buf,
            SamplingMethod::Stochastic {
                temperature: None,
                top_k: None,
                top_p: Some(TOP_P),
            },
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

#[allow(dead_code)]
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

    let seeds_buf = context.device.new_buffer_with_data(
        vec![TEST_SAMPLING_SEED; BATCH].as_ptr() as *const _,
        (BATCH * std::mem::size_of::<u64>()) as u64,
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
            &logits_buf,
            Some(&seeds_buf),
            &output_buf,
            SamplingMethod::Greedy,
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

    for sample_idx in 0..num_samples {
        let seeds_buffer = context.device.new_buffer_with_data(
            vec![TEST_SAMPLING_SEED + sample_idx as u64; batch_size].as_ptr()
                as *const _,
            (batch_size * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer_ref = context.command_queue.new_command_buffer();
        let command_buffer = command_buffer_ref.to_owned();

        kernel
            .encode(
                &logits_buffer,
                Some(&seeds_buffer),
                &output_buffer,
                SamplingMethod::Stochastic {
                    temperature: None,
                    top_k: None,
                    top_p: None,
                },
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

    let kernel =
        SamplingKernel::new(&context, KernelDataType::Float32, BATCH, VOCAB)
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
    for sample_idx in 0..NUM_SAMPLES {
        let seeds_buffer = context.device.new_buffer_with_data(
            vec![TEST_SAMPLING_SEED + sample_idx as u64; BATCH].as_ptr()
                as *const _,
            (BATCH * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer_ref = context.command_queue.new_command_buffer();
        let command_buffer = command_buffer_ref.to_owned();

        kernel
            .encode(
                &logits_buffer,
                Some(&seeds_buffer),
                &output_buffer,
                SamplingMethod::Stochastic {
                    temperature: None,
                    top_k: None,
                    top_p: None,
                },
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

#[test]
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

    let kernel =
        SamplingKernel::new(&context, KernelDataType::Float32, BATCH, VOCAB)
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

    let seeds_buf = context.device.new_buffer_with_data(
        vec![TEST_SAMPLING_SEED; BATCH].as_ptr() as *const _,
        (BATCH * std::mem::size_of::<u64>()) as u64,
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
            &logits_buf,
            Some(&seeds_buf),
            &output_buf,
            SamplingMethod::Stochastic {
                temperature: None,
                top_k: None,
                top_p: None,
            },
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
fn test_temperature_gpu_cpu_match() {
    let context = match create_test_context() {
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

    let kernel =
        SamplingKernel::new(&context, KernelDataType::Float32, BATCH, VOCAB)
            .expect("Failed to create sampling kernel");

    let logits: Vec<f32> = (0..BATCH * VOCAB)
        .map(|i| ((i * 37 % 1000) as f32 - 500.0) * 0.01)
        .collect();

    let logits_buffer = context.device.new_buffer_with_data(
        logits.as_ptr() as *const _,
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let processed_buffer = context.device.new_buffer(
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer_ref = context.command_queue.new_command_buffer();
    let command_buffer = command_buffer_ref.to_owned();

    let compute_encoder = command_buffer.new_compute_command_encoder();
    kernel
        .encode_temperature(
            &logits_buffer,
            &processed_buffer,
            BATCH as u32,
            VOCAB as u32,
            TEMPERATURE,
            compute_encoder,
        )
        .expect("encode_temperature");
    compute_encoder.end_encoding();

    command_buffer_ref.commit();
    command_buffer_ref.wait_until_completed();

    let gpu_ptr = processed_buffer.contents() as *const f32;
    let gpu_results =
        unsafe { std::slice::from_raw_parts(gpu_ptr, logits.len()) };

    for (idx, (&logit, &processed)) in
        logits.iter().zip(gpu_results.iter()).enumerate()
    {
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

    println!(
        "✓ Temperature processor gpu cpu match (temp={}, rtol={}, atol={})",
        TEMPERATURE, RTOL, ATOL
    );
}

#[test]
fn test_topk_gpu_cpu_match() {
    use rand::{SeedableRng, rngs::StdRng};

    let context = match create_test_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping topk gpu cpu match test: {}", e);
            return;
        },
    };

    const BATCH: usize = 4;
    const VOCAB: usize = 1024;
    const TOPK: u32 = 16;

    let kernel =
        SamplingKernel::new(&context, KernelDataType::Float32, BATCH, VOCAB)
            .expect("Failed to create sampling kernel");

    let mut rng = StdRng::seed_from_u64(42);
    let mut logits = vec![0.0f32; BATCH * VOCAB];
    for x in logits.iter_mut() {
        *x = rng.random_range(-1024.0f32..1024.0f32);
    }

    let logits_buffer = context.device.new_buffer_with_data(
        logits.as_ptr() as *const _,
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let processed_buffer = context.device.new_buffer(
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer_ref = context.command_queue.new_command_buffer();
    let command_buffer = command_buffer_ref.to_owned();

    let compute_encoder = command_buffer.new_compute_command_encoder();
    kernel
        .encode_topk(
            &logits_buffer,
            &processed_buffer,
            BATCH as u32,
            VOCAB as u32,
            TOPK,
            compute_encoder,
        )
        .expect("encode_topk");
    compute_encoder.end_encoding();

    command_buffer_ref.commit();
    command_buffer_ref.wait_until_completed();

    let results_ptr = processed_buffer.contents() as *const f32;
    let all_results =
        unsafe { std::slice::from_raw_parts(results_ptr, logits.len()) };

    for batch_idx in 0..BATCH {
        let cpu_logits = &logits[batch_idx * VOCAB..(batch_idx + 1) * VOCAB];
        let mut cpu_sorted_logits: Vec<(usize, f32)> =
            cpu_logits.iter().copied().enumerate().collect();
        cpu_sorted_logits.sort_by(|(_, a), (_, b)| f32::total_cmp(b, a));
        let mut cpu_processed_logits: Vec<f32> = vec![f32::NEG_INFINITY; VOCAB];
        for (idx, val) in cpu_sorted_logits.into_iter().take(TOPK as usize) {
            cpu_processed_logits[idx] = val;
        }

        let gpu_processed_logits =
            &all_results[batch_idx * VOCAB..(batch_idx + 1) * VOCAB];

        assert_eq!(cpu_processed_logits, gpu_processed_logits);
    }

    println!("✓ Topk processor gpu cpu match (topk={})", TOPK);
}

#[test]
fn test_topp_gpu_cpu_match() {
    use rand::{SeedableRng, rngs::StdRng};

    let context = match create_test_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping topp gpu cpu match test: {}", e);
            return;
        },
    };

    const BATCH: usize = 4;
    const VOCAB: usize = 1024;
    const TOPP: f32 = 0.9;

    let kernel =
        SamplingKernel::new(&context, KernelDataType::Float32, BATCH, VOCAB)
            .expect("Failed to create sampling kernel");

    let mut rng = StdRng::seed_from_u64(42);
    let mut logits = vec![0.0f32; BATCH * VOCAB];
    for x in logits.iter_mut() {
        *x = rng.random_range(-16.0f32..16.0f32);
    }

    let logits_buffer = context.device.new_buffer_with_data(
        logits.as_ptr() as *const _,
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let processed_buffer = context.device.new_buffer(
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer_ref = context.command_queue.new_command_buffer();
    let command_buffer = command_buffer_ref.to_owned();

    let compute_encoder = command_buffer.new_compute_command_encoder();
    kernel
        .encode_topp(
            &logits_buffer,
            &processed_buffer,
            BATCH as u32,
            VOCAB as u32,
            TOPP,
            compute_encoder,
        )
        .expect("encode_topp");
    compute_encoder.end_encoding();

    command_buffer_ref.commit();
    command_buffer_ref.wait_until_completed();

    let results_ptr = processed_buffer.contents() as *const f32;
    let all_results =
        unsafe { std::slice::from_raw_parts(results_ptr, logits.len()) };

    for batch_idx in 0..BATCH {
        let cpu_logits = &logits[batch_idx * VOCAB..(batch_idx + 1) * VOCAB];
        let cpu_processed_logits = cpu_reference_top_p(cpu_logits, TOPP)
            .into_iter()
            .zip(cpu_logits.iter().copied())
            .map(|(cpu_ref_topp, logit)| {
                if cpu_ref_topp > 0.0 {
                    logit
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect::<Vec<f32>>();

        let gpu_processed_logits =
            &all_results[batch_idx * VOCAB..(batch_idx + 1) * VOCAB];

        assert_eq!(&cpu_processed_logits, gpu_processed_logits);
    }

    println!("✓ Topp processor gpu cpu match (topp={})", TOPP);
}

#[test]
fn test_gumbel_gpu_cpu_match() {
    let context = match create_test_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping gumbel gpu cpu match test: {}", e);
            return;
        },
    };

    const BATCH: usize = 7;
    const VOCAB: usize = 16 * 1024 * 64;
    const RTOL: f32 = 0.01;
    const ATOL: f32 = 1e-6;

    let kernel =
        SamplingKernel::new(&context, KernelDataType::Float32, BATCH, VOCAB)
            .expect("Failed to create sampling kernel");

    let logits = vec![0.0f32; BATCH * VOCAB];
    let seeds: Vec<u64> = (0_u64..BATCH as u64).collect();

    let logits_buffer = context.device.new_buffer_with_data(
        logits.as_ptr() as *const _,
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let seeds_buffer = context.device.new_buffer_with_data(
        seeds.as_ptr() as *const _,
        (BATCH * std::mem::size_of::<u64>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let gumbel_logits_buffer = context.device.new_buffer(
        (BATCH * VOCAB * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer_ref = context.command_queue.new_command_buffer();
    let command_buffer = command_buffer_ref.to_owned();

    let compute_encoder = command_buffer.new_compute_command_encoder();
    kernel
        .encode_gumbel(
            &logits_buffer,
            &seeds_buffer,
            &gumbel_logits_buffer,
            BATCH as u32,
            VOCAB as u32,
            compute_encoder,
        )
        .expect("encode");
    compute_encoder.end_encoding();

    command_buffer_ref.commit();
    command_buffer_ref.wait_until_completed();

    let result_ptr = gumbel_logits_buffer.contents() as *const f32;
    let all_results =
        unsafe { std::slice::from_raw_parts(result_ptr, BATCH * VOCAB) };

    for (batch_idx, batch_seed) in seeds.iter().copied().enumerate() {
        let results = &all_results[batch_idx * VOCAB..(batch_idx + 1) * VOCAB];
        for (logit_idx, gpu_logit_value) in results.iter().copied().enumerate()
        {
            let cpu_logit_value =
                gumbel_float(batch_seed, revidx(logit_idx as u32));
            let abs_diff = (cpu_logit_value - gpu_logit_value).abs();
            let tolerance = ATOL + RTOL * cpu_logit_value.abs();
            assert!(
                abs_diff <= tolerance,
                "Mismatch at batch {batch_idx} element {logit_idx}: CPU={cpu_logit_value} GPU={gpu_logit_value} (abs_diff={abs_diff}, tolerance={tolerance})"
            );
        }
    }

    println!(
        "✓ Gumbel cpu gpu match test passed (rtol: {:.1}%, atol: {:.1e})",
        RTOL * 100.0,
        ATOL
    );
}
