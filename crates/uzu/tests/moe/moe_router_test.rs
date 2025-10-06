use half::bf16;
use metal::MTLResourceOptions;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::MTLContext;

fn create_ctx() -> MTLContext {
    let device = metal::Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    MTLContext::new(device, queue).expect("ctx")
}

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
            logits[token_idx * e + expert_idx] = logit;
        }
    }
    logits
}

#[test]
fn test_router_simple() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xA0073);

    // Start with minimal size to verify kernel works at all
    let t = 2;
    let e = 4;
    let d_model = 64; // Small, safe size

    eprintln!("[RouterTest] T={}, E={}, d_model={}", t, e, d_model);

    // Generate random data
    let x: Vec<bf16> = (0..t * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
        .collect();
    let weight: Vec<bf16> = (0..e * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let bias: Vec<bf16> =
        (0..e).map(|_| bf16::from_f32(rng.random_range(-0.1..0.1))).collect();

    // CPU reference
    let logits_cpu = router_cpu_reference(&x, &weight, &bias, t, e, d_model);

    // GPU buffers
    let x_buf = ctx.device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let weight_buf = ctx.device.new_buffer_with_data(
        weight.as_ptr() as *const _,
        (weight.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let bias_buf = ctx.device.new_buffer_with_data(
        bias.as_ptr() as *const _,
        (bias.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let logits_buf = ctx.device.new_buffer(
        (t * e * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Execute
    let cb = ctx.command_queue.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();

    let pipeline = ctx
        .compute_pipeline_state("moe_router_bf16", None)
        .expect("router kernel not found");
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&x_buf), 0);
    encoder.set_buffer(1, Some(&weight_buf), 0);
    encoder.set_buffer(2, Some(&bias_buf), 0);
    encoder.set_buffer(3, Some(&logits_buf), 0);

    let t_u32 = t as u32;
    let d_model_u32 = d_model as u32;
    let e_u32 = e as u32;
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &t_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &d_model_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        &e_u32 as *const u32 as *const _,
    );

    // MV-style dispatch: tile E across simdgroups per TG (S=4)
    let num_simdgroups: u64 = 4;
    let tg_x = (e as u64 + num_simdgroups - 1) / num_simdgroups;
    encoder.dispatch_thread_groups(
        metal::MTLSize::new(tg_x, t as u64, 1),
        metal::MTLSize::new(32 * num_simdgroups, 1, 1),
    );
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    // Read GPU output
    let logits_gpu = unsafe {
        std::slice::from_raw_parts(logits_buf.contents() as *const bf16, t * e)
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

    // Router matmul: error scales with sqrt(d_model) accumulation depth
    let threshold = 0.002 * (d_model as f32).sqrt();
    assert!(
        mean_abs_error < threshold,
        "Router mean_abs={:.6} exceeds threshold={:.6} for d_model={}",
        mean_abs_error,
        threshold,
        d_model
    );

    eprintln!("[RouterTest] ✓ PASSED (threshold={:.6})", threshold);
}
