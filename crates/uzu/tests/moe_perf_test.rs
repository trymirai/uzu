#![cfg(feature = "moe_dev_tests")]

use std::time::Instant;

use half::bf16;
use metal::MTLResourceOptions;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::moe::{
        MoeBucketCountsArguments, MoeBucketCountsKernel, MoeExpertsArguments,
        MoeExpertsKernel, MoeFinalizeArguments, MoeFinalizeKernel,
        MoeGatherArguments, MoeGatherKernel, MoeOffsetsScanArguments,
        MoeOffsetsScanKernel, MoeScatterKernels, MoeScatterWithMapArguments,
        MoeTopKArguments, MoeTopKKernel, RouterEncoderArgs, encode_moe_router,
        encode_moe_router_with_pipeline,
    },
};

fn create_ctx() -> MTLContext {
    let device = metal::Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    MTLContext::new(device, queue).expect("ctx")
}

struct PerfResult {
    name: String,
    mean_ms: f64,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    std_dev_ms: f64,
}

impl PerfResult {
    fn new(
        name: String,
        times_ms: &[f64],
    ) -> Self {
        let mut sorted = times_ms.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let median = sorted[sorted.len() / 2];
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        let variance = sorted.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
            / sorted.len() as f64;
        let std_dev = variance.sqrt();

        Self {
            name,
            mean_ms: mean,
            median_ms: median,
            min_ms: min,
            max_ms: max,
            std_dev_ms: std_dev,
        }
    }

    fn print(&self) {
        eprintln!(
            "  {:<30} mean={:8.3}ms  median={:8.3}ms  min={:8.3}ms  max={:8.3}ms  std={:6.3}ms",
            self.name,
            self.mean_ms,
            self.median_ms,
            self.min_ms,
            self.max_ms,
            self.std_dev_ms
        );
    }
}

fn time_kernel<F>(
    name: &str,
    warmup: usize,
    iterations: usize,
    mut f: F,
) -> PerfResult
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..warmup {
        f();
    }

    // Measure
    let mut times_ms = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed();
        times_ms.push(elapsed.as_secs_f64() * 1000.0);
    }

    PerfResult::new(name.to_string(), &times_ms)
}

// ============================================================================
// DECODE PERFORMANCE (T=1) - Latency-Critical
// ============================================================================

// Test router kernel performance in decode mode (T=1)
#[test]
fn test_router_decode_perf() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xA0073);

    let configs = vec![
        ("Small", 1, 8, 1024),
        ("Medium", 1, 16, 2048),
        ("Large", 1, 16, 4096),
        ("Production", 1, 16, 4096),
    ];

    eprintln!("\n=== Router Kernel Performance (DECODE, T=1) ===");

    for (name, t, e, d_model) in configs {
        let x: Vec<bf16> = (0..t * d_model)
            .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
            .collect();
        let weight: Vec<bf16> = (0..e * d_model)
            .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
            .collect();
        let bias: Vec<bf16> = (0..e)
            .map(|_| bf16::from_f32(rng.random_range(-0.1..0.1)))
            .collect();

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

        let perf = time_kernel(
            &format!("{} (T={}, E={}, D={})", name, t, e, d_model),
            5,
            20,
            || {
                let cb = ctx.command_queue.new_command_buffer();
                encode_moe_router(
                    &ctx,
                    &cb,
                    KernelDataType::BFloat16,
                    RouterEncoderArgs {
                        input_buffer: &x_buf,
                        weight_buffer: &weight_buf,
                        bias_buffer: &bias_buf,
                        output_buffer: &logits_buf,
                        t,
                        d_model,
                        e,
                    },
                )
                .expect("router encode failed");
                cb.commit();
                cb.wait_until_completed();
            },
        );

        perf.print();

        // For decode (T=1), show latency in microseconds
        let latency_us = (perf.mean_ms / t as f64) * 1000.0;
        eprintln!("    → Latency: {:.1} µs/token", latency_us);
    }
}

// Test router kernel performance in prefill mode (T>1)
#[test]
fn test_router_prefill_perf() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xA0073);

    let configs = vec![
        ("Batch4", 4, 16, 4096),
        ("Batch8", 8, 16, 4096),
        ("Batch16", 16, 16, 4096),
        ("Batch32", 32, 16, 4096),
        ("Batch64", 64, 16, 4096),
        ("Batch128", 128, 16, 4096),
    ];

    eprintln!("\n=== Router Kernel Performance (PREFILL, T>1) ===");

    for (name, t, e, d_model) in configs {
        let x: Vec<bf16> = (0..t * d_model)
            .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
            .collect();
        let weight: Vec<bf16> = (0..e * d_model)
            .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
            .collect();
        let bias: Vec<bf16> = (0..e)
            .map(|_| bf16::from_f32(rng.random_range(-0.1..0.1)))
            .collect();

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

        let pipeline = ctx
            .compute_pipeline_state("moe_router_bf16", None)
            .expect("router kernel not found");

        let perf = time_kernel(&format!("{} (T={})", name, t), 5, 20, || {
            let cb = ctx.command_queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
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

            let num_simdgroups: u64 = 8; // Optimized: 8 simdgroups per TG (256 threads)
            let tg_x = (e as u64 + num_simdgroups - 1) / num_simdgroups;
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(tg_x, t as u64, 1),
                metal::MTLSize::new(32 * num_simdgroups, 1, 1),
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        perf.print();

        // For prefill, show throughput
        let tokens_per_sec = (t as f64 / perf.mean_ms) * 1000.0;
        eprintln!(
            "    → Throughput: {:.1} tokens/sec, {:.3} ms/token",
            tokens_per_sec,
            perf.mean_ms / t as f64
        );
    }
}

// ============================================================================
// PREFILL PERFORMANCE (T>1) - Throughput-Oriented
// ============================================================================

// Test TopK kernel performance in decode mode (T=1)
#[test]
fn test_topk_decode_perf() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xA0074);

    let configs = vec![
        ("Small_K2", 1, 8, 2),
        ("Medium_K2", 1, 16, 2),
        ("Production_K2", 1, 16, 2),
        ("Production_K4", 1, 16, 4),
    ];

    eprintln!("\n=== TopK Kernel Performance (DECODE, T=1) ===");

    for (name, t, e, k) in configs {
        let logits: Vec<bf16> = (0..t * e)
            .map(|_| bf16::from_f32(rng.random_range(-5.0..5.0)))
            .collect();

        let logits_buf = ctx.device.new_buffer_with_data(
            logits.as_ptr() as *const _,
            (logits.len() * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let topk_ids_buf = ctx.device.new_buffer(
            (t * k * std::mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let topk_probs_buf = ctx.device.new_buffer(
            (t * k * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pipeline = ctx
            .compute_pipeline_state("moe_topk_select_bf16", None)
            .expect("topk kernel not found");

        let perf = time_kernel(
            &format!("{} (T={}, E={}, K={})", name, t, e, k),
            5,
            20,
            || {
                let cb = ctx.command_queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&logits_buf), 0);
                encoder.set_buffer(1, Some(&topk_ids_buf), 0);
                encoder.set_buffer(2, Some(&topk_probs_buf), 0);

                let t_u32 = t as u32;
                let e_u32 = e as u32;
                let k_u32 = k as u32;
                let renorm_u32 = 1u32;
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &t_u32 as *const u32 as *const _,
                );
                encoder.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &e_u32 as *const u32 as *const _,
                );
                encoder.set_bytes(
                    5,
                    std::mem::size_of::<u32>() as u64,
                    &k_u32 as *const u32 as *const _,
                );
                encoder.set_bytes(
                    6,
                    std::mem::size_of::<u32>() as u64,
                    &renorm_u32 as *const u32 as *const _,
                );

                encoder.dispatch_thread_groups(
                    metal::MTLSize::new(t as u64, 1, 1),
                    metal::MTLSize::new(256, 1, 1),
                );
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            },
        );

        perf.print();

        // For decode (T=1), show latency in nanoseconds (TopK is very fast)
        let latency_us = (perf.mean_ms / t as f64) * 1000.0;
        eprintln!("    → Latency: {:.1} µs/token", latency_us);
    }
}

// Test TopK kernel performance in prefill mode (T>1)
#[test]
fn test_topk_prefill_perf() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xA0074);

    let configs = vec![
        ("Batch4_K2", 4, 16, 2),
        ("Batch16_K2", 16, 16, 2),
        ("Batch32_K2", 32, 16, 2),
        ("Batch64_K2", 64, 16, 2),
        ("Batch128_K2", 128, 16, 2),
    ];

    eprintln!("\n=== TopK Kernel Performance (PREFILL, T>1) ===");

    for (name, t, e, k) in configs {
        let logits: Vec<bf16> = (0..t * e)
            .map(|_| bf16::from_f32(rng.random_range(-5.0..5.0)))
            .collect();

        let logits_buf = ctx.device.new_buffer_with_data(
            logits.as_ptr() as *const _,
            (logits.len() * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let topk_ids_buf = ctx.device.new_buffer(
            (t * k * std::mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let topk_probs_buf = ctx.device.new_buffer(
            (t * k * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pipeline = ctx
            .compute_pipeline_state("moe_topk_select_bf16", None)
            .expect("topk kernel not found");

        let perf = time_kernel(&format!("{} (T={})", name, t), 5, 20, || {
            let cb = ctx.command_queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&logits_buf), 0);
            encoder.set_buffer(1, Some(&topk_ids_buf), 0);
            encoder.set_buffer(2, Some(&topk_probs_buf), 0);

            let t_u32 = t as u32;
            let e_u32 = e as u32;
            let k_u32 = k as u32;
            let renorm_u32 = 1u32;
            encoder.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &t_u32 as *const u32 as *const _,
            );
            encoder.set_bytes(
                4,
                std::mem::size_of::<u32>() as u64,
                &e_u32 as *const u32 as *const _,
            );
            encoder.set_bytes(
                5,
                std::mem::size_of::<u32>() as u64,
                &k_u32 as *const u32 as *const _,
            );
            encoder.set_bytes(
                6,
                std::mem::size_of::<u32>() as u64,
                &renorm_u32 as *const u32 as *const _,
            );

            encoder.dispatch_thread_groups(
                metal::MTLSize::new(t as u64, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        perf.print();

        // For prefill, show throughput
        let tokens_per_sec = (t as f64 / perf.mean_ms) * 1000.0;
        eprintln!(
            "    → Throughput: {:.1} tokens/sec, {:.3} µs/token",
            tokens_per_sec,
            (perf.mean_ms / t as f64) * 1000.0
        );
    }
}

// Test E2E MoE performance with timing breakdown (decode mode, T=1)
#[test]
fn test_moe_e2e_decode_perf() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xA0075);

    let configs = vec![
        ("Small", 1, 1024, 256, 8, 2),
        ("Medium", 1, 2048, 1024, 16, 2),
        ("Production", 1, 4096, 14336, 16, 2),
    ];

    eprintln!("\n=== End-to-End MoE Performance (DECODE, T=1) ===");

    for (config_name, t, d_model, d_ff, e, k) in configs {
        eprintln!(
            "\n{}  T={}, D={}, H={}, E={}, K={}",
            config_name, t, d_model, d_ff, e, k
        );

        // Generate data
        let x: Vec<bf16> = (0..t * d_model)
            .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
            .collect();
        let router_w: Vec<bf16> = (0..e * d_model)
            .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
            .collect();
        let router_b: Vec<bf16> = (0..e)
            .map(|_| bf16::from_f32(rng.random_range(-0.1..0.1)))
            .collect();

        // Buffers (simplified - just key buffers for timing)
        let x_buf = ctx.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let router_w_buf = ctx.device.new_buffer_with_data(
            router_w.as_ptr() as *const _,
            (router_w.len() * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let router_b_buf = ctx.device.new_buffer_with_data(
            router_b.as_ptr() as *const _,
            (router_b.len() * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let logits_buf = ctx.device.new_buffer(
            (t * e * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let topk_ids_buf = ctx.device.new_buffer(
            (t * k * std::mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let topk_probs_buf = ctx.device.new_buffer(
            (t * k * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let router_pipeline =
            ctx.compute_pipeline_state("moe_router_bf16", None).unwrap();
        let topk_pipeline =
            ctx.compute_pipeline_state("moe_topk_select_bf16", None).unwrap();

        // Time router
        let router_perf = time_kernel("Router", 5, 20, || {
            let cb = ctx.command_queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&router_pipeline);
            encoder.set_buffer(0, Some(&x_buf), 0);
            encoder.set_buffer(1, Some(&router_w_buf), 0);
            encoder.set_buffer(2, Some(&router_b_buf), 0);
            encoder.set_buffer(3, Some(&logits_buf), 0);

            let t_u32 = t as u32;
            let d_u32 = d_model as u32;
            let e_u32 = e as u32;
            encoder.set_bytes(4, 4, &t_u32 as *const u32 as *const _);
            encoder.set_bytes(5, 4, &d_u32 as *const u32 as *const _);
            encoder.set_bytes(6, 4, &e_u32 as *const u32 as *const _);

            let num_sg: u64 = 4;
            let tg_x = (e as u64 + num_sg - 1) / num_sg;
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(tg_x, t as u64, 1),
                metal::MTLSize::new(32 * num_sg, 1, 1),
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Time TopK
        let topk_perf = time_kernel("TopK", 5, 20, || {
            let cb = ctx.command_queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&topk_pipeline);
            encoder.set_buffer(0, Some(&logits_buf), 0);
            encoder.set_buffer(1, Some(&topk_ids_buf), 0);
            encoder.set_buffer(2, Some(&topk_probs_buf), 0);

            let t_u32 = t as u32;
            let e_u32 = e as u32;
            let k_u32 = k as u32;
            let renorm = 1u32;
            encoder.set_bytes(3, 4, &t_u32 as *const u32 as *const _);
            encoder.set_bytes(4, 4, &e_u32 as *const u32 as *const _);
            encoder.set_bytes(5, 4, &k_u32 as *const u32 as *const _);
            encoder.set_bytes(6, 4, &renorm as *const u32 as *const _);

            encoder.dispatch_thread_groups(
                metal::MTLSize::new(t as u64, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        router_perf.print();
        topk_perf.print();

        let total_ms = router_perf.mean_ms + topk_perf.mean_ms;
        let router_pct = (router_perf.mean_ms / total_ms) * 100.0;
        let topk_pct = (topk_perf.mean_ms / total_ms) * 100.0;

        eprintln!("\n  Breakdown:");
        eprintln!("    Router:  {:6.2}%", router_pct);
        eprintln!("    TopK:    {:6.2}%", topk_pct);
        eprintln!("    Total:   {:8.1} µs/token", total_ms * 1000.0);
    }
}

// Test E2E MoE performance with timing breakdown (prefill mode, T>1)
#[test]
fn test_moe_e2e_prefill_perf() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xA0076);

    let configs = vec![
        ("Batch4", 4, 4096, 14336, 16, 2),
        ("Batch16", 16, 4096, 14336, 16, 2),
        ("Batch32", 32, 4096, 14336, 16, 2),
        ("Batch64", 64, 4096, 14336, 16, 2),
    ];

    eprintln!("\n=== End-to-End MoE Performance (PREFILL, T>1) ===");

    for (config_name, t, d_model, d_ff, e, k) in configs {
        eprintln!(
            "\n{}  T={}, D={}, H={}, E={}, K={}",
            config_name, t, d_model, d_ff, e, k
        );

        // Generate data
        let x: Vec<bf16> = (0..t * d_model)
            .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
            .collect();
        let router_w: Vec<bf16> = (0..e * d_model)
            .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
            .collect();
        let router_b: Vec<bf16> = (0..e)
            .map(|_| bf16::from_f32(rng.random_range(-0.1..0.1)))
            .collect();

        // Buffers (simplified - just key buffers for timing)
        let x_buf = ctx.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let router_w_buf = ctx.device.new_buffer_with_data(
            router_w.as_ptr() as *const _,
            (router_w.len() * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let router_b_buf = ctx.device.new_buffer_with_data(
            router_b.as_ptr() as *const _,
            (router_b.len() * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let logits_buf = ctx.device.new_buffer(
            (t * e * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let topk_ids_buf = ctx.device.new_buffer(
            (t * k * std::mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let topk_probs_buf = ctx.device.new_buffer(
            (t * k * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let router_pipeline =
            ctx.compute_pipeline_state("moe_router_bf16", None).unwrap();
        let topk_pipeline =
            ctx.compute_pipeline_state("moe_topk_select_bf16", None).unwrap();

        // Time router
        let router_perf = time_kernel("Router", 5, 20, || {
            let cb = ctx.command_queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&router_pipeline);
            encoder.set_buffer(0, Some(&x_buf), 0);
            encoder.set_buffer(1, Some(&router_w_buf), 0);
            encoder.set_buffer(2, Some(&router_b_buf), 0);
            encoder.set_buffer(3, Some(&logits_buf), 0);

            let t_u32 = t as u32;
            let d_u32 = d_model as u32;
            let e_u32 = e as u32;
            encoder.set_bytes(4, 4, &t_u32 as *const u32 as *const _);
            encoder.set_bytes(5, 4, &d_u32 as *const u32 as *const _);
            encoder.set_bytes(6, 4, &e_u32 as *const u32 as *const _);

            let num_sg: u64 = 4;
            let tg_x = (e as u64 + num_sg - 1) / num_sg;
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(tg_x, t as u64, 1),
                metal::MTLSize::new(32 * num_sg, 1, 1),
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Time TopK
        let topk_perf = time_kernel("TopK", 5, 20, || {
            let cb = ctx.command_queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&topk_pipeline);
            encoder.set_buffer(0, Some(&logits_buf), 0);
            encoder.set_buffer(1, Some(&topk_ids_buf), 0);
            encoder.set_buffer(2, Some(&topk_probs_buf), 0);

            let t_u32 = t as u32;
            let e_u32 = e as u32;
            let k_u32 = k as u32;
            let renorm = 1u32;
            encoder.set_bytes(3, 4, &t_u32 as *const u32 as *const _);
            encoder.set_bytes(4, 4, &e_u32 as *const u32 as *const _);
            encoder.set_bytes(5, 4, &k_u32 as *const u32 as *const _);
            encoder.set_bytes(6, 4, &renorm as *const u32 as *const _);

            encoder.dispatch_thread_groups(
                metal::MTLSize::new(t as u64, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        router_perf.print();
        topk_perf.print();

        let total_ms = router_perf.mean_ms + topk_perf.mean_ms;
        let router_pct = (router_perf.mean_ms / total_ms) * 100.0;
        let topk_pct = (topk_perf.mean_ms / total_ms) * 100.0;

        eprintln!("\n  Breakdown:");
        eprintln!("    Router:  {:6.2}%", router_pct);
        eprintln!("    TopK:    {:6.2}%", topk_pct);
        eprintln!("    Total:   {:8.3} ms", total_ms);
        eprintln!(
            "    Throughput: {:.1} tokens/sec, {:.3} ms/token",
            (t as f64 / total_ms) * 1000.0,
            total_ms / t as f64
        );
    }
}

// Test complete MoE pipeline timing breakdown (decode mode, T=1)
#[test]
fn test_moe_pipeline_breakdown_decode() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xDECADE);

    eprintln!("\n=== MoE Pipeline Breakdown (DECODE, T=1) ===");
    eprintln!(
        "Measures ALL MoE kernels: Router→TopK→Counts→Offsets→Scatter→Gather→Experts→Finalize\n"
    );

    let (t, d_model, d_ff, e, k) = (1, 4096, 14336, 16, 2);

    // Allocate buffers
    let x: Vec<bf16> = (0..t * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
        .collect();
    let router_w: Vec<bf16> = (0..e * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let router_b: Vec<bf16> =
        (0..e).map(|_| bf16::from_f32(rng.random_range(-0.1..0.1))).collect();

    let x_buf = ctx.device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let router_w_buf = ctx.device.new_buffer_with_data(
        router_w.as_ptr() as *const _,
        (router_w.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let router_b_buf = ctx.device.new_buffer_with_data(
        router_b.as_ptr() as *const _,
        (router_b.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let logits_buf = ctx.device.new_buffer(
        (t * e * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let topk_ids_buf = ctx.device.new_buffer(
        (t * k * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let topk_probs_buf = ctx.device.new_buffer(
        (t * k * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let counts_buf = ctx.device.new_buffer(
        (e * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let num_blocks = ((t + 255) / 256).max(1);
    let num_tiles = ((e + 512 - 1) / 512).max(1);
    let partials_buf = ctx.device.new_buffer(
        (num_blocks * num_tiles * e * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let offsets_buf = ctx.device.new_buffer(
        ((e + 1) * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sumk_buf = ctx.device.new_buffer(
        std::mem::size_of::<i32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let bucketed_ids_buf = ctx.device.new_buffer(
        (t * k * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let x_perm_buf = ctx.device.new_buffer(
        (t * k * d_model * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_partial_buf = ctx.device.new_buffer(
        (t * k * d_model * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let tok2row_buf = ctx.device.new_buffer(
        (t * k * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_out_buf = ctx.device.new_buffer(
        (t * d_model * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Expert weights buffers
    let w13: Vec<bf16> = (0..e * d_model * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let w2: Vec<bf16> = (0..e * d_ff * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.random_range(-0.1..0.1)))
        .collect();
    let down_biases: Vec<bf16> = (0..e * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.1..0.1)))
        .collect();

    let w13_buf = ctx.device.new_buffer_with_data(
        w13.as_ptr() as *const _,
        (w13.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w2_buf = ctx.device.new_buffer_with_data(
        w2.as_ptr() as *const _,
        (w2.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let up_biases_buf = ctx.device.new_buffer_with_data(
        up_biases.as_ptr() as *const _,
        (up_biases.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let down_biases_buf = ctx.device.new_buffer_with_data(
        down_biases.as_ptr() as *const _,
        (down_biases.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Experts tiling buffers
    const BN: usize = 64;
    let num_tiles_n = (d_model + BN - 1) / BN;
    let max_tiles = t * k * e * num_tiles_n;
    let tile_counts_buf = ctx.device.new_buffer(
        (e * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let tile_offsets_buf = ctx.device.new_buffer(
        ((e + 1) * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let tile_map_buf = ctx.device.new_buffer(
        (max_tiles * 3 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let total_tiles_buf = ctx.device.new_buffer(
        (8 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let dispatch_args_buf = ctx.device.new_buffer(
        (3 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Scatter block bases buffers
    let block_bases_buf = ctx.device.new_buffer(
        (num_blocks * num_tiles * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let block_alloc_buf = ctx.device.new_buffer(
        (num_blocks * num_tiles * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let bucketed_probs_buf = ctx.device.new_buffer(
        (t * k * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create kernel structs (use production-validated encoding logic)
    let router_pipeline = ctx
        .compute_pipeline_state("moe_router_bf16", None)
        .expect("router pipeline");
    let topk_kernel = MoeTopKKernel::new(&ctx).expect("topk");
    let counts_kernel = MoeBucketCountsKernel::new(&ctx).expect("counts");
    let offsets_kernel = MoeOffsetsScanKernel::new(&ctx).expect("offsets");
    let scatter_kernel = MoeScatterKernels::new(&ctx).expect("scatter");
    let gather_kernel = MoeGatherKernel::new(&ctx).expect("gather");
    let experts_kernel = MoeExpertsKernel::new(&ctx).expect("experts");
    let finalize_kernel = MoeFinalizeKernel::new(&ctx).expect("finalize");

    // Time each kernel (reduced iterations for isolation)
    let router_perf = time_kernel("Router", 2, 5, || {
        let cb = ctx.command_queue.new_command_buffer();
        encode_moe_router_with_pipeline(
            &router_pipeline,
            &cb,
            RouterEncoderArgs {
                input_buffer: &x_buf,
                weight_buffer: &router_w_buf,
                bias_buffer: &router_b_buf,
                output_buffer: &logits_buf,
                t,
                d_model,
                e,
            },
        );
        cb.commit();
        cb.wait_until_completed();
    });

    let topk_perf = time_kernel("TopK", 2, 5, || {
        let cb = ctx.command_queue.new_command_buffer();
        topk_kernel
            .encode(
                &cb,
                KernelDataType::BFloat16,
                MoeTopKArguments {
                    logits_buffer: &logits_buf,
                    topk_ids_buffer: &topk_ids_buf,
                    topk_probs_buffer: &topk_probs_buf,
                    t,
                    e,
                    k,
                    renorm: true,
                },
            )
            .expect("topk");
        cb.commit();
        cb.wait_until_completed();
    });

    // Testing: Router + TopK + Counts
    let counts_perf = time_kernel("Counts", 2, 5, || {
        let cb = ctx.command_queue.new_command_buffer();
        counts_kernel
            .encode(
                &cb,
                MoeBucketCountsArguments {
                    topk_ids_buffer: &topk_ids_buf,
                    counts_buffer: &counts_buf,
                    partials_buffer: &partials_buf,
                    t,
                    e,
                    k,
                },
            )
            .expect("counts");
        cb.commit();
        cb.wait_until_completed();
    });

    let offsets_perf = time_kernel("Offsets", 2, 5, || {
        let cb = ctx.command_queue.new_command_buffer();
        offsets_kernel
            .encode(
                &cb,
                MoeOffsetsScanArguments {
                    counts_buffer: &counts_buf,
                    offsets_buffer: &offsets_buf,
                    sumk_buffer: &sumk_buf,
                    e,
                },
            )
            .expect("offsets");
        cb.commit();
        cb.wait_until_completed();
    });

    let scatter_perf = time_kernel("Scatter", 2, 5, || {
        let cb = ctx.command_queue.new_command_buffer();
        scatter_kernel
            .encode_block_bases(
                &cb,
                uzu::backends::metal::kernel::moe::MoeBlockBasesArguments {
                    partials_buffer: &partials_buf,
                    block_bases_buffer: &block_bases_buf,
                    block_alloc_buffer: &block_alloc_buf,
                    e,
                    num_blocks,
                    num_tiles,
                },
            )
            .expect("block bases");
        scatter_kernel
            .encode_scatter_with_map(
                &cb,
                MoeScatterWithMapArguments {
                    base:
                        uzu::backends::metal::kernel::moe::MoeScatterArguments {
                            topk_ids_buffer: &topk_ids_buf,
                            topk_probs_buffer: &topk_probs_buf,
                            offsets_buffer: &offsets_buf,
                            block_bases_buffer: &block_bases_buf,
                            block_alloc_buffer: &block_alloc_buf,
                            out_ids_buffer: &bucketed_ids_buf,
                            out_probs_buffer: &bucketed_probs_buf,
                            t,
                            e,
                            k,
                            num_blocks,
                            num_tiles,
                        },
                    tok2row_buffer: &tok2row_buf,
                },
                KernelDataType::BFloat16,
            )
            .expect("scatter");
        cb.commit();
        cb.wait_until_completed();
    });

    let gather_perf = time_kernel("Gather", 2, 5, || {
        let cb = ctx.command_queue.new_command_buffer();
        gather_kernel
            .encode(
                &cb,
                KernelDataType::BFloat16,
                MoeGatherArguments {
                    x_buffer: &x_buf,
                    bucketed_ids_buffer: &bucketed_ids_buf,
                    x_perm_buffer: &x_perm_buf,
                    sumk_buffer: &sumk_buf,
                    t,
                    k,
                    d_model,
                },
            )
            .expect("gather");
        cb.commit();
        cb.wait_until_completed();
    });

    let experts_perf = time_kernel("Experts (MAIN COMPUTE)", 2, 5, || {
        let cb = ctx.command_queue.new_command_buffer();
        experts_kernel
            .encode(
                &cb,
                MoeExpertsArguments {
                    x_perm_buffer: &x_perm_buf,
                    expert_offsets: &offsets_buf,
                    w13_all: &w13_buf,
                    w2_all: &w2_buf,
                    y_partial: &y_partial_buf,
                    up_biases: &up_biases_buf,
                    down_biases: &down_biases_buf,
                    tile_counts: &tile_counts_buf,
                    tile_row_offsets: &tile_offsets_buf,
                    tile_map: &tile_map_buf,
                    total_tiles: &total_tiles_buf,
                    dispatch_args: &dispatch_args_buf,
                    num_tiles_n,
                    t,
                    d_model,
                    d_ff,
                    e,
                    k,
                    gating_code: 2, // SILU
                    gate_clip_min: f32::NEG_INFINITY,
                    gate_clip_max: 20.0,
                    up_clip_min: -19.0,
                    up_clip_max: 21.0,
                    silu_alpha: 1.702,
                    data_type: KernelDataType::BFloat16,
                },
            )
            .expect("experts");
        cb.commit();
        cb.wait_until_completed();
    });

    let finalize_perf = time_kernel("Finalize", 2, 5, || {
        let cb = ctx.command_queue.new_command_buffer();
        finalize_kernel
            .encode(
                &cb,
                MoeFinalizeArguments {
                    tok2row_buffer: &tok2row_buf,
                    probs_buffer: &topk_probs_buf,
                    y_partial_buffer: &y_partial_buf,
                    y_out_buffer: &y_out_buf,
                    t,
                    d_model,
                    k,
                },
                KernelDataType::BFloat16,
            )
            .expect("finalize");
        cb.commit();
        cb.wait_until_completed();
    });

    // Print results
    router_perf.print();
    topk_perf.print();
    counts_perf.print();
    offsets_perf.print();
    scatter_perf.print();
    gather_perf.print();
    experts_perf.print();
    finalize_perf.print();

    // Calculate breakdown
    let total_us = (router_perf.mean_ms
        + topk_perf.mean_ms
        + counts_perf.mean_ms
        + offsets_perf.mean_ms
        + scatter_perf.mean_ms
        + gather_perf.mean_ms
        + experts_perf.mean_ms
        + finalize_perf.mean_ms)
        * 1000.0;

    eprintln!(
        "\n  ═══ Per-Kernel Latency (Production D=4096, H=14336, E=16, K=2, T=1) ═══"
    );
    eprintln!(
        "    Router:   {:8.1} us  ({:5.1}%)",
        router_perf.mean_ms * 1000.0,
        (router_perf.mean_ms / (total_us / 1000.0)) * 100.0
    );
    eprintln!(
        "    TopK:     {:8.1} us  ({:5.1}%)",
        topk_perf.mean_ms * 1000.0,
        (topk_perf.mean_ms / (total_us / 1000.0)) * 100.0
    );
    eprintln!(
        "    Counts:   {:8.1} us  ({:5.1}%)",
        counts_perf.mean_ms * 1000.0,
        (counts_perf.mean_ms / (total_us / 1000.0)) * 100.0
    );
    eprintln!(
        "    Offsets:  {:8.1} us  ({:5.1}%)",
        offsets_perf.mean_ms * 1000.0,
        (offsets_perf.mean_ms / (total_us / 1000.0)) * 100.0
    );
    eprintln!(
        "    Scatter:  {:8.1} us  ({:5.1}%)",
        scatter_perf.mean_ms * 1000.0,
        (scatter_perf.mean_ms / (total_us / 1000.0)) * 100.0
    );
    eprintln!(
        "    Gather:   {:8.1} us  ({:5.1}%)",
        gather_perf.mean_ms * 1000.0,
        (gather_perf.mean_ms / (total_us / 1000.0)) * 100.0
    );
    eprintln!(
        "    Experts:  {:8.1} us  ({:5.1}%) ← MAIN COMPUTE",
        experts_perf.mean_ms * 1000.0,
        (experts_perf.mean_ms / (total_us / 1000.0)) * 100.0
    );
    eprintln!(
        "    Finalize: {:8.1} us  ({:5.1}%)",
        finalize_perf.mean_ms * 1000.0,
        (finalize_perf.mean_ms / (total_us / 1000.0)) * 100.0
    );
    eprintln!("    ═══════════════════════════════════════════");
    eprintln!("    TOTAL:    {:8.1} us (100.0%)", total_us);
    eprintln!(
        "\n  Note: Times include Metal CB overhead (~10-50ms per kernel)."
    );
    eprintln!(
        "        Real GPU compute is much faster, but relative % shows bottleneck."
    );
}
