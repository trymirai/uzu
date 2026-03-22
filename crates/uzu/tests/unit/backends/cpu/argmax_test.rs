use std::time::Instant;

use super::*;

#[cfg(target_arch = "aarch64")]
#[test]
fn test_neon_argmax_correctness() {
    // Single deterministic 64k array test with random values
    const SIZE: usize = 64 * 1024;
    const SEED: u64 = 12345;

    // Simple deterministic random number generator
    let mut rng_state = SEED;
    let mut data = Vec::with_capacity(SIZE);

    for _ in 0..SIZE {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let random_f32 = (rng_state as f32) / (u64::MAX as f32);
        data.push(random_f32);
    }

    // Set a known max at position 42_000
    data[42_000] = f32::MAX;

    // Compute expected result with naive method
    let expected = simple_argmax(&data);
    // Compare NEON result to expected
    let neon_result = neon_optimized_argmax(&data);

    assert_eq!(expected, neon_result, "64k array test failed: expected {}, got {}", expected, neon_result);
    assert_eq!(expected, 42_000, "Expected max should be at index 42_000");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_neon_performance_128k() {
    const SEQ_LEN: usize = 128 * 1024;
    let batch_size = 2;
    const SEED: u64 = 67890;

    // Generate random data with deterministic seed
    let mut rng_state = SEED;
    let mut data = Vec::with_capacity(batch_size * SEQ_LEN);

    for _ in 0..(batch_size * SEQ_LEN) {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let random_f32 = (rng_state as f32) / (u64::MAX as f32);
        data.push(random_f32);
    }

    // Set known max positions
    let expected_indices = vec![42_000, 100_000];
    for (batch_idx, &max_pos) in expected_indices.iter().enumerate() {
        let offset = batch_idx * SEQ_LEN + max_pos;
        data[offset] = f32::MAX;
    }

    // Test each batch separately
    for (batch_idx, &expected_idx) in expected_indices.iter().enumerate() {
        let start_offset = batch_idx * SEQ_LEN;
        let end_offset = start_offset + SEQ_LEN;
        let batch_data = &data[start_offset..end_offset];

        // Test with NEON implementation
        let start = Instant::now();
        let neon_result = neon_optimized_argmax(batch_data);
        let neon_duration = start.elapsed();

        // Test with simple implementation
        let start = Instant::now();
        let simple_result = simple_argmax(batch_data);
        let simple_duration = start.elapsed();

        // Test with optimized implementation (manual unrolling)
        let start = Instant::now();
        let optimized_result = optimized_argmax(batch_data);
        let optimized_duration = start.elapsed();

        if batch_idx == 0 {
            println!("128k argmax performance comparison:");
            println!("NEON:      {:.3} ms", neon_duration.as_secs_f64() * 1000.0);
            println!("Simple:    {:.3} ms", simple_duration.as_secs_f64() * 1000.0);
            println!("Optimized: {:.3} ms", optimized_duration.as_secs_f64() * 1000.0);

            if simple_duration.as_nanos() > 0 && neon_duration.as_nanos() > 0 {
                let speedup_simple = simple_duration.as_secs_f64() / neon_duration.as_secs_f64();
                let speedup_optimized = optimized_duration.as_secs_f64() / neon_duration.as_secs_f64();
                println!("NEON vs Simple speedup:    {:.2}x", speedup_simple);
                println!("NEON vs Optimized speedup: {:.2}x", speedup_optimized);
            }
        }

        // Verify all methods give the same result
        assert_eq!(
            neon_result, expected_idx,
            "NEON: Batch {} argmax mismatch: expected {}, got {}",
            batch_idx, expected_idx, neon_result
        );
        assert_eq!(
            simple_result, expected_idx,
            "Simple: Batch {} argmax mismatch: expected {}, got {}",
            batch_idx, expected_idx, simple_result
        );
        assert_eq!(
            optimized_result, expected_idx,
            "Optimized: Batch {} argmax mismatch: expected {}, got {}",
            batch_idx, expected_idx, optimized_result
        );
    }
}
