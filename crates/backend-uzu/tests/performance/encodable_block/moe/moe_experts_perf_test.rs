use std::time::Instant;

use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};

use super::{MoeExpertsTwoPassArguments, MoeExpertsTwoPassDecodeBlock, MoeExpertsTwoPassPrefillBlock};
use crate::{
    backends::common::{Backend, Encoder},
    common::helpers::{alloc_allocation_with_data, create_context},
    data_type::DataType,
};

fn build_offsets(
    e: usize,
    sum_k: usize,
) -> Vec<u32> {
    let mut offsets = vec![0u32; e + 1];
    let mut running = 0u32;
    let mut remaining = sum_k;
    for expert in 0..e {
        offsets[expert] = running;
        if remaining > 0 {
            running += 1;
            remaining -= 1;
        }
    }
    offsets[e] = running;
    offsets
}

fn run_decode_case<B: Backend>(
    ctx: &B::Context,
    name: &str,
    t: usize,
    d_model: usize,
    d_ff: usize,
    e: usize,
    k: usize,
    warmup: usize,
    iters: usize,
) {
    let mut rng = StdRng::seed_from_u64(0xDEC0DE1234567890);
    let sum_k = t * k;

    eprintln!("\n[decode/two-pass] {} => T={}, D={}, FF={}, E={}, K={}, sum_k={}", name, t, d_model, d_ff, e, k, sum_k);

    let x_perm: Vec<bf16> = (0..sum_k * d_model).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect();
    let offsets = build_offsets(e, sum_k);

    // Generate W13 in original layout [E, d_model, 2*d_ff]
    let w13_original: Vec<bf16> =
        (0..e * d_model * 2 * d_ff).map(|_| bf16::from_f32(rng.random_range(-0.05..0.05))).collect();

    // Transpose to GPU layout [E, 2*d_ff, d_model]
    let mut w13 = vec![bf16::from_f32(0.0); e * d_model * 2 * d_ff];
    for expert in 0..e {
        let src_offset = expert * d_model * 2 * d_ff;
        let dst_offset = expert * 2 * d_ff * d_model;
        for dm in 0..d_model {
            for ff in 0..(2 * d_ff) {
                let src_idx = src_offset + dm * 2 * d_ff + ff;
                let dst_idx = dst_offset + ff * d_model + dm;
                w13[dst_idx] = w13_original[src_idx];
            }
        }
    }
    let w2: Vec<bf16> = (0..e * d_ff * d_model).map(|_| bf16::from_f32(rng.random_range(-0.05..0.05))).collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff).map(|_| bf16::from_f32(rng.random_range(-0.01..0.01))).collect();
    let down_biases: Vec<bf16> = (0..e * d_model).map(|_| bf16::from_f32(rng.random_range(-0.01..0.01))).collect();

    let experts_kernel = MoeExpertsTwoPassDecodeBlock::<B>::new(ctx, DataType::BF16, 2).expect("experts decode kernel");

    let x_perm_buf = alloc_allocation_with_data::<B, bf16>(ctx, &x_perm);
    let offsets_buf = alloc_allocation_with_data::<B, u32>(ctx, &offsets);
    let w13_buf = alloc_allocation_with_data::<B, bf16>(ctx, &w13);
    let w2_buf = alloc_allocation_with_data::<B, bf16>(ctx, &w2);
    let up_biases_buf = alloc_allocation_with_data::<B, bf16>(ctx, &up_biases);
    let down_biases_buf = alloc_allocation_with_data::<B, bf16>(ctx, &down_biases);

    for _ in 0..warmup {
        let mut encoder = Encoder::new(ctx).expect("Failed to create encoder");
        let output = experts_kernel
            .encode(
                MoeExpertsTwoPassArguments {
                    x_perm: &x_perm_buf,
                    expert_offsets: &offsets_buf,
                    w13_all: &w13_buf,
                    w2_all: &w2_buf,
                    up_biases: &up_biases_buf,
                    down_biases: &down_biases_buf,
                    total_rows: sum_k,
                    d_model,
                    d_ff,
                    num_routed_experts: e,
                    gate_clip_min: f32::NEG_INFINITY,
                    gate_clip_max: f32::INFINITY,
                    up_clip_min: f32::NEG_INFINITY,
                    up_clip_max: f32::INFINITY,
                    silu_alpha: 1.702,
                },
                &mut encoder,
            )
            .expect("failed to encode MoE experts");
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        drop(output);
        drop(completed);
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let mut encoder = Encoder::new(ctx).expect("Failed to create encoder");
        let output = experts_kernel
            .encode(
                MoeExpertsTwoPassArguments {
                    x_perm: &x_perm_buf,
                    expert_offsets: &offsets_buf,
                    w13_all: &w13_buf,
                    w2_all: &w2_buf,
                    up_biases: &up_biases_buf,
                    down_biases: &down_biases_buf,
                    total_rows: sum_k,
                    d_model,
                    d_ff,
                    num_routed_experts: e,
                    gate_clip_min: f32::NEG_INFINITY,
                    gate_clip_max: f32::INFINITY,
                    up_clip_min: f32::NEG_INFINITY,
                    up_clip_max: f32::INFINITY,
                    silu_alpha: 1.702,
                },
                &mut encoder,
            )
            .expect("failed to encode MoE experts");
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        drop(output);
        drop(completed);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    let var = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let std = var.sqrt();

    eprintln!("    mean={:.3}ms median={:.3}ms min={:.3}ms max={:.3}ms std={:.3}ms", mean, median, min, max, std);
    eprintln!("    → Latency: {:.1} µs/token", (mean / t as f64) * 1000.0);
}

fn run_two_pass_prefill_case<B: Backend>(
    ctx: &B::Context,
    name: &str,
    t: usize,
    d_model: usize,
    d_ff: usize,
    e: usize,
    k: usize,
    warmup: usize,
    iters: usize,
) {
    let mut rng = StdRng::seed_from_u64(0xDEC0DE1234567890);
    let sum_k = t * k;

    eprintln!("\n[2-pass prefill] {} => T={}, D={}, FF={}, E={}, K={}, sum_k={}", name, t, d_model, d_ff, e, k, sum_k);

    let x_perm: Vec<bf16> = (0..sum_k * d_model).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect();
    let offsets = build_offsets(e, sum_k);

    let w13_original: Vec<bf16> =
        (0..e * d_model * 2 * d_ff).map(|_| bf16::from_f32(rng.random_range(-0.05..0.05))).collect();

    let mut w13 = vec![bf16::from_f32(0.0); e * d_model * 2 * d_ff];
    for expert in 0..e {
        let src_offset = expert * d_model * 2 * d_ff;
        let dst_offset = expert * 2 * d_ff * d_model;
        for dm in 0..d_model {
            for ff in 0..(2 * d_ff) {
                let src_idx = src_offset + dm * 2 * d_ff + ff;
                let dst_idx = dst_offset + ff * d_model + dm;
                w13[dst_idx] = w13_original[src_idx];
            }
        }
    }
    let w2: Vec<bf16> = (0..e * d_ff * d_model).map(|_| bf16::from_f32(rng.random_range(-0.05..0.05))).collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff).map(|_| bf16::from_f32(rng.random_range(-0.01..0.01))).collect();
    let down_biases: Vec<bf16> = (0..e * d_model).map(|_| bf16::from_f32(rng.random_range(-0.01..0.01))).collect();

    let experts_kernel =
        MoeExpertsTwoPassPrefillBlock::<B>::new(ctx, DataType::BF16, 2).expect("experts prefill kernel");

    let x_perm_buf = alloc_allocation_with_data::<B, bf16>(ctx, &x_perm);
    let offsets_buf = alloc_allocation_with_data::<B, u32>(ctx, &offsets);
    let w13_buf = alloc_allocation_with_data::<B, bf16>(ctx, &w13);
    let w2_buf = alloc_allocation_with_data::<B, bf16>(ctx, &w2);
    let up_biases_buf = alloc_allocation_with_data::<B, bf16>(ctx, &up_biases);
    let down_biases_buf = alloc_allocation_with_data::<B, bf16>(ctx, &down_biases);

    for _ in 0..warmup {
        let mut encoder = Encoder::new(ctx).expect("Failed to create encoder");
        let args = MoeExpertsTwoPassArguments {
            x_perm: &x_perm_buf,
            expert_offsets: &offsets_buf,
            w13_all: &w13_buf,
            w2_all: &w2_buf,
            up_biases: &up_biases_buf,
            down_biases: &down_biases_buf,
            total_rows: sum_k,
            d_model,
            d_ff,
            num_routed_experts: e,
            gate_clip_min: f32::NEG_INFINITY,
            gate_clip_max: f32::INFINITY,
            up_clip_min: f32::NEG_INFINITY,
            up_clip_max: f32::INFINITY,
            silu_alpha: 1.702,
        };
        let output = experts_kernel.encode(args, &mut encoder).expect("failed to encode MoE experts");
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        drop(output);
        drop(completed);
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let mut encoder = Encoder::new(ctx).expect("Failed to create encoder");
        let args = MoeExpertsTwoPassArguments {
            x_perm: &x_perm_buf,
            expert_offsets: &offsets_buf,
            w13_all: &w13_buf,
            w2_all: &w2_buf,
            up_biases: &up_biases_buf,
            down_biases: &down_biases_buf,
            total_rows: sum_k,
            d_model,
            d_ff,
            num_routed_experts: e,
            gate_clip_min: f32::NEG_INFINITY,
            gate_clip_max: f32::INFINITY,
            up_clip_min: f32::NEG_INFINITY,
            up_clip_max: f32::INFINITY,
            silu_alpha: 1.702,
        };
        let output = experts_kernel.encode(args, &mut encoder).expect("failed to encode MoE experts");
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        drop(output);
        drop(completed);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    let var = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let std = var.sqrt();

    eprintln!("    mean={:.3}ms median={:.3}ms min={:.3}ms max={:.3}ms std={:.3}ms", mean, median, min, max, std);
    eprintln!("    → Throughput: {:.1} µs/token (mean / sum_k)", (mean / sum_k as f64) * 1000.0);
}

#[test]
#[ignore]
fn test_two_pass_decode_speed() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();

        let cases = vec![
            ("T1_E16_D1024_F4096", 1, 1024, 4096, 16, 4, 3, 20),
            ("T1_E8_D2048_F8192", 1, 2048, 8192, 8, 4, 3, 20),
        ];

        for (name, t, d_model, d_ff, e, k, warmup, iters) in &cases {
            run_decode_case::<B>(&ctx, name, *t, *d_model, *d_ff, *e, *k, *warmup, *iters);
        }
    });
}

#[test]
#[ignore]
fn test_two_pass_prefill_speed() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();

        let cases = vec![
            ("T32_E16_D1024_F4096", 32, 1024, 4096, 16, 4, 2, 10),
            ("T64_E16_D2048_F6144", 64, 2048, 6144, 16, 4, 2, 10),
            ("T256_E16_D2048_F6144", 256, 2048, 6144, 16, 4, 2, 10),
        ];

        for (name, t, d_model, d_ff, e, k, warmup, iters) in cases {
            run_two_pass_prefill_case::<B>(&ctx, name, t, d_model, d_ff, e, k, warmup, iters);
        }
    })
}
