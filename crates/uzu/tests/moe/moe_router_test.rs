use half::bf16;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::kernel::{
    KernelDataType,
    moe::{MoeRouterArguments, MoeRouterKernel},
};

use super::test_utils::{alloc_buffer, alloc_buffer_with_data, create_ctx};

fn router_cpu_reference(
    x: &[bf16],      // [T, d_model]
    weight: &[bf16], // [E, d_model]
    bias: &[bf16],   // [E]
    t: usize,
    e: usize,
    d_model: usize,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; t * e];
    for token_idx in 0..t {
        for expert_idx in 0..e {
            let mut logit = f32::from(bias[expert_idx]);
            for d in 0..d_model {
                let x_val = f32::from(x[token_idx * d_model + d]);
                let w_val = f32::from(weight[expert_idx * d_model + d]);
                logit += x_val * w_val;
            }
            logits[token_idx * e + expert_idx] =
                f32::from(bf16::from_f32(logit));
        }
    }
    logits
}

#[test]
fn test_router_correctness() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xA0073);

    // Test multiple shapes: (T, E, d_model)
    let shapes = vec![
        (1, 4, 64),    // Single token, small
        (2, 4, 64),    // Minimal batch
        (8, 8, 128),   // Balanced
        (32, 16, 256), // Larger batch
        (1, 64, 512),  // Many experts
        (64, 8, 1024), // Large model
    ];

    for (t, e, d_model) in shapes {
        eprintln!("[RouterTest] T={}, E={}, d_model={}", t, e, d_model);

        // Generate random data
        let x: Vec<bf16> = (0..t * d_model)
            .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
            .collect();
        let weight: Vec<bf16> = (0..e * d_model)
            .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
            .collect();
        let bias: Vec<bf16> = (0..e)
            .map(|_| bf16::from_f32(rng.random_range(-0.1..0.1)))
            .collect();

        // CPU reference
        let logits_cpu =
            router_cpu_reference(&x, &weight, &bias, t, e, d_model);

        // GPU buffers
        let x_buf = alloc_buffer_with_data(&ctx, &x);
        let weight_buf = alloc_buffer_with_data(&ctx, &weight);
        let bias_buf = alloc_buffer_with_data(&ctx, &bias);
        let logits_buf = alloc_buffer::<bf16>(&ctx, t * e);

        // Execute using kernel struct API
        let router = MoeRouterKernel::new(&ctx).expect("MoeRouterKernel::new");
        let cb = ctx.command_queue.new_command_buffer();
        router
            .encode(
                &cb,
                KernelDataType::BFloat16,
                MoeRouterArguments {
                    input_buffer: &x_buf,
                    weight_buffer: &weight_buf,
                    bias_buffer: &bias_buf,
                    output_buffer: &logits_buf,
                    t,
                    d_model,
                    e,
                },
            )
            .expect("encode router");
        cb.commit();
        cb.wait_until_completed();

        // Read GPU output
        let logits_gpu = unsafe {
            std::slice::from_raw_parts(
                logits_buf.contents() as *const bf16,
                t * e,
            )
        };

        // Compare
        let mut max_abs_diff = 0.0f32;
        let mut total_abs_error = 0.0f32;
        for i in 0..(t * e) {
            let gpu_val = f32::from(logits_gpu[i]);
            let cpu_val = logits_cpu[i];
            let abs_diff = (gpu_val - cpu_val).abs();
            max_abs_diff = max_abs_diff.max(abs_diff);
            total_abs_error += abs_diff;

            if abs_diff > 0.01 {
                eprintln!(
                    "[RouterTest] Mismatch [{}]: GPU={:.6}, CPU={:.6}, diff={:.6}",
                    i, gpu_val, cpu_val, abs_diff
                );
            }
        }
        let mean_abs_error = total_abs_error / (t * e) as f32;

        eprintln!(
            "[RouterTest] Error: max_abs={:.6}, mean_abs={:.6}",
            max_abs_diff, mean_abs_error
        );

        let threshold = 0.001;
        assert!(
            max_abs_diff < threshold,
            "Router max_abs={:.6} exceeds threshold={:.6} for d_model={}",
            max_abs_diff,
            threshold,
            d_model
        );

        eprintln!(
            "[RouterTest] ✓ PASSED (max_err={:.6}, threshold={:.6})",
            max_abs_diff, threshold
        );
    }
}
