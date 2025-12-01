#![cfg(target_os = "macos")]

//! Performance test comparing conditional vs non-conditional execution in async batch mode.
//!
//! This test measures the overhead of GPU-side conditional execution and the savings
//! when preconditions are invalidated to skip remaining work.
//!
//! Environment variables:
//! - UZU_CONDITIONAL_EXECUTION: "1" (default) or "0" to enable/disable conditional execution
//! - UZU_BATCH_SIZE: batch size for async mode (default 128)
//! - UZU_ASYNC_MODE: "batch" (default), "lookahead", or "sync"

mod common;

use std::time::Instant;

use uzu::session::{
    Session,
    config::{DecodingConfig, RunConfig},
    parameter::SamplingSeed,
    types::{Input, Output},
};

fn build_decoding_config() -> DecodingConfig {
    DecodingConfig::default().with_sampling_seed(SamplingSeed::Custom(42))
}

/// Test configuration
struct TestConfig {
    /// Number of tokens to generate
    tokens_to_generate: u64,
    /// Number of warmup iterations
    warmup_iterations: usize,
    /// Number of measurement iterations
    measure_iterations: usize,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            tokens_to_generate: 64,
            warmup_iterations: 2,
            measure_iterations: 5,
        }
    }
}

/// Results from a benchmark run
#[derive(Debug, Clone)]
struct BenchmarkResult {
    avg_duration: f64,
    avg_tokens: usize,
    tokens_per_second: f64,
}

/// Run benchmark with specified settings
fn run_benchmark(
    model_path: &std::path::Path,
    prompt: &str,
    tokens_to_generate: u64,
    stop_after: Option<usize>,
    warmup: usize,
    iterations: usize,
) -> BenchmarkResult {
    let mut session =
        Session::new(model_path.to_path_buf(), build_decoding_config())
            .expect("Failed to create session");

    // Warmup
    for _ in 0..warmup {
        if let Some(stop) = stop_after {
            let _ = measure_generation_with_early_stop(
                &mut session,
                prompt,
                tokens_to_generate,
                stop,
            );
        } else {
            let _ =
                measure_generation(&mut session, prompt, tokens_to_generate);
        }
        let _ = session.reset();
    }

    // Measure
    let mut durations = Vec::with_capacity(iterations);
    let mut total_tokens = 0;

    for _ in 0..iterations {
        let (duration, tokens) = if let Some(stop) = stop_after {
            measure_generation_with_early_stop(
                &mut session,
                prompt,
                tokens_to_generate,
                stop,
            )
        } else {
            measure_generation(&mut session, prompt, tokens_to_generate)
        };
        durations.push(duration);
        total_tokens += tokens;
        let _ = session.reset();
    }

    let avg_duration = durations.iter().sum::<f64>() / durations.len() as f64;
    let avg_tokens = total_tokens / iterations;
    let tokens_per_second = avg_tokens as f64 / avg_duration;

    BenchmarkResult {
        avg_duration,
        avg_tokens,
        tokens_per_second,
    }
}

/// Run generation and measure time
fn measure_generation(
    session: &mut Session,
    prompt: &str,
    tokens_limit: u64,
) -> (f64, usize) {
    let input = Input::Text(prompt.to_string());

    let start = Instant::now();
    let output = session
        .run(
            input,
            RunConfig::default().tokens_limit(tokens_limit),
            None::<fn(Output) -> bool>,
        )
        .expect("Generation failed");
    let duration = start.elapsed().as_secs_f64();

    let tokens_generated =
        output.stats.total_stats.tokens_count_output as usize;
    (duration, tokens_generated)
}

/// Run generation with early stop (simulates stop condition detection)
fn measure_generation_with_early_stop(
    session: &mut Session,
    prompt: &str,
    tokens_limit: u64,
    stop_after: usize,
) -> (f64, usize) {
    let input = Input::Text(prompt.to_string());
    let tokens_received = std::sync::atomic::AtomicUsize::new(0);

    let start = Instant::now();
    let output = session
        .run(
            input,
            RunConfig::default().tokens_limit(tokens_limit),
            Some(|_output: Output| {
                let count = tokens_received
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Return false to stop after `stop_after` tokens
                count < stop_after
            }),
        )
        .expect("Generation failed");
    let duration = start.elapsed().as_secs_f64();

    let tokens_generated =
        output.stats.total_stats.tokens_count_output as usize;
    (duration, tokens_generated)
}

#[test]
#[ignore] // Run with: cargo test conditional_vs_unconditional -- --ignored --nocapture
fn conditional_vs_unconditional_comparison() {
    let model_path = common::get_test_model_path();

    println!("\n=== Conditional vs Non-Conditional Execution Comparison ===\n");
    println!("Model: {:?}", model_path);
    println!(
        "This test compares GPU-side conditional execution (enabled vs disabled)"
    );
    println!("when generation stops early.\n");

    let prompt = "Explain the concept of machine learning in simple terms.";
    let tokens_to_generate = 128u64;
    let warmup = 2;
    let iterations = 5;

    // Test scenarios with different early stop points
    let stop_scenarios = [
        (None, "Full generation (no early stop)"),
        (Some(64), "Stop at 64 tokens (50%)"),
        (Some(32), "Stop at 32 tokens (25%)"),
        (Some(10), "Stop at 10 tokens (very early)"),
    ];

    for (stop_after, description) in stop_scenarios {
        println!("\n--- {} ---", description);

        // Run with conditional execution DISABLED
        // SAFETY: Test runs in isolation
        unsafe { std::env::set_var("UZU_CONDITIONAL_EXECUTION", "0") };
        let result_disabled = run_benchmark(
            &model_path,
            prompt,
            tokens_to_generate,
            stop_after,
            warmup,
            iterations,
        );
        println!(
            "  Without conditional: {:.3}s for {} tokens ({:.1} tok/s)",
            result_disabled.avg_duration,
            result_disabled.avg_tokens,
            result_disabled.tokens_per_second
        );

        // Run with conditional execution ENABLED
        // SAFETY: Test runs in isolation
        unsafe { std::env::set_var("UZU_CONDITIONAL_EXECUTION", "1") };
        let result_enabled = run_benchmark(
            &model_path,
            prompt,
            tokens_to_generate,
            stop_after,
            warmup,
            iterations,
        );
        println!(
            "  With conditional:    {:.3}s for {} tokens ({:.1} tok/s)",
            result_enabled.avg_duration,
            result_enabled.avg_tokens,
            result_enabled.tokens_per_second
        );

        // Calculate difference
        let time_diff =
            result_disabled.avg_duration - result_enabled.avg_duration;
        let time_diff_pct = (time_diff / result_disabled.avg_duration) * 100.0;
        let speedup = result_enabled.tokens_per_second
            / result_disabled.tokens_per_second;

        if time_diff > 0.0 {
            println!(
                "  → Conditional saved {:.3}s ({:.1}% faster, {:.2}x speedup)",
                time_diff, time_diff_pct, speedup
            );
        } else {
            println!(
                "  → Conditional overhead: {:.3}s ({:.1}% slower)",
                -time_diff, -time_diff_pct
            );
        }
    }

    // Reset env var
    // SAFETY: Test runs in isolation
    unsafe { std::env::remove_var("UZU_CONDITIONAL_EXECUTION") };

    println!("\n=== Test Complete ===\n");
    println!("Notes:");
    println!(
        "  - Conditional execution has minimal overhead for full generation"
    );
    println!("  - Savings increase when generation stops earlier");
    println!("  - Batch size affects the amount of skippable work");
}

#[test]
#[ignore] // Run with: cargo test conditional_execution_perf -- --ignored --nocapture
fn conditional_execution_perf_comparison() {
    let model_path = common::get_test_model_path();
    let config = TestConfig::default();

    println!("\n=== Conditional Execution Performance Test ===\n");
    println!("Model: {:?}", model_path);
    println!("Tokens to generate: {}", config.tokens_to_generate);
    println!("Warmup iterations: {}", config.warmup_iterations);
    println!("Measurement iterations: {}", config.measure_iterations);

    let prompt = "Explain the concept of machine learning in simple terms.";

    // Test 1: Full generation (no early stop)
    println!("\n--- Test 1: Full Generation (no early stop) ---");
    {
        let mut session =
            Session::new(model_path.clone(), build_decoding_config())
                .expect("Failed to create session");

        // Warmup
        for i in 0..config.warmup_iterations {
            let (duration, tokens) = measure_generation(
                &mut session,
                prompt,
                config.tokens_to_generate,
            );
            println!(
                "  Warmup {}: {:.3}s for {} tokens",
                i + 1,
                duration,
                tokens
            );
            let _ = session.reset();
        }

        // Measure
        let mut durations = Vec::with_capacity(config.measure_iterations);
        let mut total_tokens = 0;

        for i in 0..config.measure_iterations {
            let (duration, tokens) = measure_generation(
                &mut session,
                prompt,
                config.tokens_to_generate,
            );
            println!(
                "  Run {}: {:.3}s for {} tokens ({:.1} tok/s)",
                i + 1,
                duration,
                tokens,
                tokens as f64 / duration
            );
            durations.push(duration);
            total_tokens += tokens;
            let _ = session.reset();
        }

        let avg_duration =
            durations.iter().sum::<f64>() / durations.len() as f64;
        let avg_tokens = total_tokens / config.measure_iterations;
        println!(
            "\n  Average: {:.3}s for {} tokens ({:.1} tok/s)",
            avg_duration,
            avg_tokens,
            avg_tokens as f64 / avg_duration
        );
    }

    // Test 2: Generation with early stop (triggers precondition invalidation)
    println!("\n--- Test 2: Generation with Early Stop ---");
    let stop_after_tokens = (config.tokens_to_generate / 2) as usize;
    println!("  Stopping after {} tokens", stop_after_tokens);
    {
        let mut session =
            Session::new(model_path.clone(), build_decoding_config())
                .expect("Failed to create session");

        // Warmup
        for i in 0..config.warmup_iterations {
            let (duration, tokens) = measure_generation_with_early_stop(
                &mut session,
                prompt,
                config.tokens_to_generate,
                stop_after_tokens,
            );
            println!(
                "  Warmup {}: {:.3}s for {} tokens",
                i + 1,
                duration,
                tokens
            );
            let _ = session.reset();
        }

        // Measure
        let mut durations = Vec::with_capacity(config.measure_iterations);
        let mut total_tokens = 0;

        for i in 0..config.measure_iterations {
            let (duration, tokens) = measure_generation_with_early_stop(
                &mut session,
                prompt,
                config.tokens_to_generate,
                stop_after_tokens,
            );
            println!(
                "  Run {}: {:.3}s for {} tokens ({:.1} tok/s)",
                i + 1,
                duration,
                tokens,
                tokens as f64 / duration
            );
            durations.push(duration);
            total_tokens += tokens;
            let _ = session.reset();
        }

        let avg_duration =
            durations.iter().sum::<f64>() / durations.len() as f64;
        let avg_tokens = total_tokens / config.measure_iterations;
        println!(
            "\n  Average: {:.3}s for {} tokens ({:.1} tok/s)",
            avg_duration,
            avg_tokens,
            avg_tokens as f64 / avg_duration
        );
    }

    // Test 3: Very early stop (more wasted work to skip)
    println!("\n--- Test 3: Very Early Stop (10 tokens) ---");
    let very_early_stop = 10;
    {
        let mut session =
            Session::new(model_path.clone(), build_decoding_config())
                .expect("Failed to create session");

        // Warmup
        for i in 0..config.warmup_iterations {
            let (duration, tokens) = measure_generation_with_early_stop(
                &mut session,
                prompt,
                config.tokens_to_generate,
                very_early_stop,
            );
            println!(
                "  Warmup {}: {:.3}s for {} tokens",
                i + 1,
                duration,
                tokens
            );
            let _ = session.reset();
        }

        // Measure
        let mut durations = Vec::with_capacity(config.measure_iterations);
        let mut total_tokens = 0;

        for i in 0..config.measure_iterations {
            let (duration, tokens) = measure_generation_with_early_stop(
                &mut session,
                prompt,
                config.tokens_to_generate,
                very_early_stop,
            );
            println!(
                "  Run {}: {:.3}s for {} tokens ({:.1} tok/s)",
                i + 1,
                duration,
                tokens,
                tokens as f64 / duration
            );
            durations.push(duration);
            total_tokens += tokens;
            let _ = session.reset();
        }

        let avg_duration =
            durations.iter().sum::<f64>() / durations.len() as f64;
        let avg_tokens = total_tokens / config.measure_iterations;
        println!(
            "\n  Average: {:.3}s for {} tokens ({:.1} tok/s)",
            avg_duration,
            avg_tokens,
            avg_tokens as f64 / avg_duration
        );
    }

    println!("\n=== Test Complete ===\n");
    println!(
        "Note: The conditional execution optimization reduces GPU work when"
    );
    println!("generation stops early. The savings depend on:");
    println!("  - Batch size (UZU_BATCH_SIZE env var, default 128)");
    println!("  - How early the stop occurs within a batch");
    println!("  - Model size and layer complexity");
}

#[test]
#[ignore] // Run with: cargo test batch_size_comparison -- --ignored --nocapture
fn batch_size_comparison() {
    let model_path = common::get_test_model_path();

    println!("\n=== Batch Size Comparison Test ===\n");
    println!("Model: {:?}", model_path);

    let prompt = "Write a short story about a robot learning to paint.";
    let tokens_to_generate = 128u64;
    let stop_after = 20usize;

    for batch_size in [16, 32, 64, 128] {
        println!("\n--- Batch Size: {} ---", batch_size);

        // Set environment variable for batch size
        // SAFETY: This test is run in isolation and batch size is valid
        unsafe { std::env::set_var("UZU_BATCH_SIZE", batch_size.to_string()) };

        let mut session =
            Session::new(model_path.clone(), build_decoding_config())
                .expect("Failed to create session");

        // Warmup
        let _ = measure_generation_with_early_stop(
            &mut session,
            prompt,
            tokens_to_generate,
            stop_after,
        );
        let _ = session.reset();

        // Measure 3 times
        let mut durations = Vec::new();
        for _ in 0..3 {
            let (duration, tokens) = measure_generation_with_early_stop(
                &mut session,
                prompt,
                tokens_to_generate,
                stop_after,
            );
            durations.push((duration, tokens));
            let _ = session.reset();
        }

        let avg_duration: f64 =
            durations.iter().map(|(d, _)| d).sum::<f64>() / 3.0;
        let avg_tokens: f64 =
            durations.iter().map(|(_, t)| *t as f64).sum::<f64>() / 3.0;

        println!(
            "  Stop after {} tokens, batch_size={}",
            stop_after, batch_size
        );
        println!(
            "  Average: {:.3}s for {:.0} tokens ({:.1} tok/s)",
            avg_duration,
            avg_tokens,
            avg_tokens / avg_duration
        );
        println!(
            "  Potential wasted batches: {} (batch_size - (stop_after % batch_size))",
            batch_size - (stop_after % batch_size)
        );
    }

    // Reset env var
    // SAFETY: This test is run in isolation
    unsafe { std::env::remove_var("UZU_BATCH_SIZE") };

    println!("\n=== Test Complete ===\n");
}
