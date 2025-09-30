#![cfg(any(target_os = "macos", target_os = "ios"))]

use std::time::Instant;

use half::f16;
use metal::{Device, MTLResourceOptions};
use rand::SeedableRng;
use uzu::backends::metal::{
    MTLContext,
    kernel::{MoeExpertsArguments, MoeExpertsKernel},
};

fn create_ctx() -> MTLContext {
    let d = Device::system_default().expect("no metal");
    let q = d.new_command_queue();
    MTLContext::new(d, q).expect("ctx")
}

fn run_case(
    ctx: &MTLContext,
    t: usize,
    e: usize,
    d_model: usize,
    d_ff: usize,
    gating_code: u32,
    iters: usize,
) {
    // Inputs
    let mut rng = rand::rngs::StdRng::seed_from_u64(2025);
    let x: Vec<f16> = (0..t * d_model)
        .map(|_| f16::from_f32(rand::Rng::random_range(&mut rng, -1.0..1.0)))
        .collect();

    // Build offsets (uniform) and bucket ids (sequential)
    let sum_k = t; // top-1 equivalent mapping
    let mut offsets = vec![0u32; e + 1];
    for i in 0..e {
        offsets[i + 1] = (((i + 1) * sum_k) / e) as u32;
    }
    let bucket_ids: Vec<i32> = (0..sum_k as i32).collect();

    // Weights
    let w1: Vec<f16> = (0..e * d_ff * d_model)
        .map(|_| f16::from_f32(rand::Rng::random_range(&mut rng, -0.5..0.5)))
        .collect();
    let w2: Vec<f16> = (0..e * d_model * d_ff)
        .map(|_| f16::from_f32(rand::Rng::random_range(&mut rng, -0.5..0.5)))
        .collect();
    let w3: Vec<f16> = (0..e * d_ff * d_model)
        .map(|_| f16::from_f32(rand::Rng::random_range(&mut rng, -0.5..0.5)))
        .collect();

    // Buffers
    let x_buf = ctx.device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let bid_buf = ctx.device.new_buffer_with_data(
        bucket_ids.as_ptr() as *const _,
        (bucket_ids.len() * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let off_buf = ctx.device.new_buffer_with_data(
        offsets.as_ptr() as *const _,
        (offsets.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w1_buf = ctx.device.new_buffer_with_data(
        w1.as_ptr() as *const _,
        (w1.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w3_buf = ctx.device.new_buffer_with_data(
        w3.as_ptr() as *const _,
        (w3.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w2_buf = ctx.device.new_buffer_with_data(
        w2.as_ptr() as *const _,
        (w2.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mut y = vec![f16::from_f32(0.0); sum_k * d_model];
    let y_buf = ctx.device.new_buffer_with_data(
        y.as_ptr() as *const _,
        (y.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let kf = MoeExpertsKernel::new(ctx).expect("experts");
    let empty_buf =
        ctx.device.new_buffer(0, metal::MTLResourceOptions::StorageModeShared);

    // Warmup
    {
        let cb = ctx.command_queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        kf.encode(
            &enc,
            MoeExpertsArguments {
                x_buffer: &x_buf,
                bucketed_token_ids: &bid_buf,
                expert_offsets: &off_buf,
                w1_all: &w1_buf,
                w3_all: &w3_buf,
                w2_all: &w2_buf,
                y_partial: &y_buf,
                up_biases: &empty_buf,
                down_biases: &empty_buf,
                t,
                d_model,
                d_ff,
                e,
                gating_code,
                gate_clip_min: f32::NEG_INFINITY,
                gate_clip_max: f32::INFINITY,
                up_clip_min: f32::NEG_INFINITY,
                up_clip_max: f32::INFINITY,
                silu_alpha: 1.0,
            },
        )
        .expect("encode experts");
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Timed iterations
    let start = Instant::now();
    for _ in 0..iters {
        let cb = ctx.command_queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        kf.encode(
            &enc,
            MoeExpertsArguments {
                x_buffer: &x_buf,
                bucketed_token_ids: &bid_buf,
                expert_offsets: &off_buf,
                w1_all: &w1_buf,
                w3_all: &w3_buf,
                w2_all: &w2_buf,
                y_partial: &y_buf,
                up_biases: &empty_buf,
                down_biases: &empty_buf,
                t,
                d_model,
                d_ff,
                e,
                gating_code,
                gate_clip_min: f32::NEG_INFINITY,
                gate_clip_max: f32::INFINITY,
                up_clip_min: f32::NEG_INFINITY,
                up_clip_max: f32::INFINITY,
                silu_alpha: 1.0,
            },
        )
        .expect("encode experts");
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let toks = (t * iters) as f64;
    let toks_per_s = toks / elapsed.max(1e-9);
    println!(
        "MoE fused perf: T={}, E={}, d_model={}, d_ff={}, gate={} -> {:.1} tok/s ({} iters, {:.3}s)",
        t, e, d_model, d_ff, gating_code, toks_per_s, iters, elapsed
    );
}

#[test]
#[ignore]
fn moe_fused_expert_mlp_perf_prefill_and_decode() {
    let ctx = create_ctx();
    // Prefill-like (throughput)
    run_case(&ctx, 1024, 8, 256, 1024, 2, 10); // SwiGLU
    run_case(&ctx, 1024, 8, 256, 1024, 0, 10); // GELU
    // Decode-like (latency)
    run_case(&ctx, 4, 8, 256, 1024, 2, 200); // more iters for stable timing
    run_case(&ctx, 1, 8, 256, 1024, 0, 500);
}
