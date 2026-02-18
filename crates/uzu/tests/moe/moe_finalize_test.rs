#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::bf16;
use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::{
    DataType,
    backends::{common::kernel::MoeFinalizeKernel, metal::kernel::dsl::MoeFinalizeMetalKernel},
};

use super::test_utils::{alloc_buffer, alloc_buffer_with_data, assert_bf16_close, create_ctx};

fn cpu_finalize(
    tok2row: &[i32],
    probs: &[bf16],
    y_partial: &[bf16],
    t: usize,
    d_model: usize,
    k: usize,
) -> Vec<bf16> {
    let mut y = vec![bf16::from_f32(0.0); t * d_model];
    for ti in 0..t {
        for f in 0..d_model {
            let mut acc = 0f32;
            for kk in 0..k {
                let idx = ti * k + kk;
                let row = tok2row[idx];
                if row >= 0 {
                    let rowu = row as usize;
                    acc += f32::from(probs[idx]) * f32::from(y_partial[rowu * d_model + f]);
                }
            }
            y[ti * d_model + f] = bf16::from_f32(acc);
        }
    }
    y
}

#[test]
fn test_finalize_correctness() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(2026);

    // Test multiple shapes: (T, K, d_model, sum_k)
    let shapes = vec![
        (1, 2, 64, 2),    // Single token, K=2
        (4, 2, 128, 8),   // Small batch
        (8, 4, 256, 32),  // Medium
        (16, 2, 512, 32), // Large d_model
    ];

    for (t, k, d_model, sum_k) in shapes {
        eprintln!("[FinalizeTest] T={}, K={}, d_model={}, sum_k={}", t, k, d_model, sum_k);

        // Generate random tok2row mapping: maps (token, k_idx) → row in y_partial
        // Some entries can be -1 (no expert selected)
        let mut tok2row: Vec<i32> = (0..t * k)
            .map(|_| {
                if rng.random_bool(0.9) {
                    rng.random_range(0..sum_k as i32)
                } else {
                    -1 // No expert selected
                }
            })
            .collect();

        // Ensure we use all rows in sum_k (avoid unused rows)
        for row in 0..sum_k.min(t * k) {
            tok2row[row] = row as i32;
        }

        // Generate random probabilities (should sum to 1 per token, but not critical for unit test)
        let probs: Vec<bf16> = (0..t * k).map(|_| bf16::from_f32(rng.random_range(0.0..1.0))).collect();

        // Generate random y_partial (expert outputs)
        let y_partial: Vec<bf16> = (0..sum_k * d_model).map(|_| bf16::from_f32(rng.random_range(-2.0..2.0))).collect();

        // CPU reference
        let y_cpu = cpu_finalize(&tok2row, &probs, &y_partial, t, d_model, k);

        // GPU buffers
        let tok2row_buf = alloc_buffer_with_data(&ctx, &tok2row);
        let probs_buf = alloc_buffer_with_data(&ctx, &probs);
        let y_partial_buf = alloc_buffer_with_data(&ctx, &y_partial);
        let y_out_buf = alloc_buffer::<bf16>(&ctx, t * d_model);

        // Execute finalize kernel
        let finalize = MoeFinalizeMetalKernel::new(&ctx, DataType::BF16).expect("finalize kernel");
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        let encoder = cb.new_compute_command_encoder().expect("encoder");
        finalize.encode(
            &tok2row_buf,
            &probs_buf,
            &y_partial_buf,
            &y_out_buf,
            t as u32,
            d_model as u32,
            k as u32,
            &encoder,
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // Compare
        let y_gpu = unsafe { std::slice::from_raw_parts(y_out_buf.contents().as_ptr() as *const bf16, t * d_model) };

        // BF16 has limited precision, especially for weighted sums
        assert_bf16_close(y_gpu, &y_cpu, 1e-2, "finalize output");

        eprintln!("[FinalizeTest] ✓ PASSED");
    }
}
