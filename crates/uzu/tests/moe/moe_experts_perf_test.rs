use std::time::Instant;

use half::bf16;
use metal::{MTLCommandBuffer, MTLCommandQueue};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::{
    DataType,
    backends::{
        common::kernel::moe::{
            MoeExpertsSingleDecodeArguments, MoeExpertsSingleDecodeKernels, MoeExpertsTwoPassArguments,
            MoeExpertsTwoPassDecodeBlock, MoeExpertsTwoPassPrefillBlock,
        },
        metal::{Metal, MetalContext},
    },
};

use super::test_utils::{alloc_buffer, alloc_buffer_with_data, create_ctx};

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

fn run_decode_case(
    ctx: &MetalContext,
    name: &str,
    t: usize,
    d_model: usize,
    d_ff: usize,
    e: usize,
    k: usize,
    warmup: usize,
    iters: usize,
) {
    const K_TILE: usize = 64;

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

    let experts_kernel = MoeExpertsTwoPassDecodeBlock::<Metal>::new(ctx).expect("experts decode kernel");

    let x_perm_buf = alloc_buffer_with_data(&ctx, &x_perm);
    let offsets_buf = alloc_buffer_with_data(&ctx, &offsets);

    let w13_buf = alloc_buffer_with_data(&ctx, &w13);

    let w2_buf = alloc_buffer_with_data(&ctx, &w2);

    let up_biases_buf = alloc_buffer_with_data(&ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data(&ctx, &down_biases);

    let num_tiles_k = ((d_ff + K_TILE - 1) / K_TILE) as usize;

    let hidden_buf = alloc_buffer::<f32>(&ctx, sum_k * d_ff);
    let output_buf = alloc_buffer::<bf16>(&ctx, sum_k * d_model);

    // Buffers for indirect dispatch
    let tile_counts_buf = alloc_buffer::<u32>(&ctx, e);
    let tile_offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let max_tiles = e * sum_k * 16384; // Conservative upper bound
    let tile_map_buf = alloc_buffer::<u32>(&ctx, max_tiles * 3);
    let total_tiles_buf = alloc_buffer::<u32>(&ctx, 1);
    let dispatch_args_buf = alloc_buffer::<u32>(&ctx, 3);
    let row_expert_map_buf = alloc_buffer::<u32>(&ctx, sum_k);

    let make_two_pass_args = || MoeExpertsTwoPassArguments {
        x_perm_buffer: &x_perm_buf,
        expert_offsets: &offsets_buf,
        row_expert_map: &row_expert_map_buf,
        hidden_buffer: &hidden_buf,
        output_buffer: &output_buf,
        w13_all: &w13_buf,
        w2_all: &w2_buf,
        up_biases: &up_biases_buf,
        down_biases: &down_biases_buf,
        tile_counts: &tile_counts_buf,
        tile_offsets: &tile_offsets_buf,
        tile_map: &tile_map_buf,
        total_tiles: &total_tiles_buf,
        dispatch_args: &dispatch_args_buf,
        total_rows: sum_k,
        d_model,
        d_ff,
        e,
        num_tiles_k: num_tiles_k as u32,
        gating_code: 2,
        gate_clip_min: f32::NEG_INFINITY,
        gate_clip_max: f32::INFINITY,
        up_clip_min: f32::NEG_INFINITY,
        up_clip_max: f32::INFINITY,
        silu_alpha: 1.702,
        data_type: DataType::BF16,
    };

    for _ in 0..warmup {
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        experts_kernel.encode(&cb, &make_two_pass_args());
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        experts_kernel.encode(&cb, &make_two_pass_args());
        cb.commit();
        cb.wait_until_completed();
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

#[test]
#[ignore]
fn test_two_pass_decode_speed() {
    let ctx = create_ctx();

    let cases =
        vec![("T1_E16_D1024_F4096", 1, 1024, 4096, 16, 4, 3, 20), ("T1_E8_D2048_F8192", 1, 2048, 8192, 8, 4, 3, 20)];

    for (name, t, d_model, d_ff, e, k, warmup, iters) in &cases {
        run_decode_case(&ctx, name, *t, *d_model, *d_ff, *e, *k, *warmup, *iters);
    }
}

fn run_two_pass_prefill_case(
    ctx: &MetalContext,
    name: &str,
    t: usize,
    d_model: usize,
    d_ff: usize,
    e: usize,
    k: usize,
    warmup: usize,
    iters: usize,
) {
    const K_TILE: usize = 64;

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

    let experts_kernel = MoeExpertsTwoPassPrefillBlock::<Metal>::new(ctx).expect("experts prefill kernel");

    let x_perm_buf = alloc_buffer_with_data(&ctx, &x_perm);
    let offsets_buf = alloc_buffer_with_data(&ctx, &offsets);
    let w13_buf = alloc_buffer_with_data(&ctx, &w13);
    let w2_buf = alloc_buffer_with_data(&ctx, &w2);
    let up_biases_buf = alloc_buffer_with_data(&ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data(&ctx, &down_biases);

    let num_tiles_k = ((d_ff + K_TILE - 1) / K_TILE) as usize;

    let hidden_buf = alloc_buffer::<f32>(&ctx, sum_k * d_ff);
    let output_buf = alloc_buffer::<bf16>(&ctx, sum_k * d_model);

    let tile_counts_buf = alloc_buffer::<u32>(&ctx, e);
    let tile_offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let max_tiles = e * sum_k * 16384;
    let tile_map_buf = alloc_buffer::<u32>(&ctx, max_tiles * 3);
    let total_tiles_buf = alloc_buffer::<u32>(&ctx, 1);
    let dispatch_args_buf = alloc_buffer::<u32>(&ctx, 3);
    let row_expert_map_buf = alloc_buffer::<u32>(&ctx, sum_k);

    let make_two_pass_args = || MoeExpertsTwoPassArguments {
        x_perm_buffer: &x_perm_buf,
        expert_offsets: &offsets_buf,
        row_expert_map: &row_expert_map_buf,
        hidden_buffer: &hidden_buf,
        output_buffer: &output_buf,
        w13_all: &w13_buf,
        w2_all: &w2_buf,
        up_biases: &up_biases_buf,
        down_biases: &down_biases_buf,
        tile_counts: &tile_counts_buf,
        tile_offsets: &tile_offsets_buf,
        tile_map: &tile_map_buf,
        total_tiles: &total_tiles_buf,
        dispatch_args: &dispatch_args_buf,
        total_rows: sum_k,
        d_model,
        d_ff,
        e,
        num_tiles_k: num_tiles_k as u32,
        gating_code: 2,
        gate_clip_min: f32::NEG_INFINITY,
        gate_clip_max: f32::INFINITY,
        up_clip_min: f32::NEG_INFINITY,
        up_clip_max: f32::INFINITY,
        silu_alpha: 1.702,
        data_type: DataType::BF16,
    };

    for _ in 0..warmup {
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        experts_kernel.encode(&cb, &make_two_pass_args());
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        experts_kernel.encode(&cb, &make_two_pass_args());
        cb.commit();
        cb.wait_until_completed();
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
fn test_two_pass_prefill_speed() {
    let ctx = create_ctx();

    let cases = vec![
        ("T32_E16_D1024_F4096", 32, 1024, 4096, 16, 4, 2, 10),
        ("T64_E16_D2048_F6144", 64, 2048, 6144, 16, 4, 2, 10),
        ("T256_E16_D2048_F6144", 256, 2048, 6144, 16, 4, 2, 10),
    ];

    for (name, t, d_model, d_ff, e, k, warmup, iters) in cases {
        run_two_pass_prefill_case(&ctx, name, t, d_model, d_ff, e, k, warmup, iters);
    }
}

fn run_fused_single_token_case(
    ctx: &MetalContext,
    name: &str,
    d_model: usize,
    d_ff: usize,
    e: usize,
    k: usize,
    warmup: usize,
    iters: usize,
) {
    let mut rng = StdRng::seed_from_u64(0xF053ED);

    eprintln!("\n[fused single-token] {} => D={}, FF={}, E={}, K={}", name, d_model, d_ff, e, k);

    let x: Vec<bf16> = (0..d_model).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect();

    let topk_ids: Vec<i32> = (0..k).map(|i| (i % e) as i32).collect();

    let topk_probs: Vec<bf16> = {
        let raw: Vec<f32> = (0..k).map(|_| rng.random_range(0.1..1.0)).collect();
        let sum: f32 = raw.iter().sum();
        raw.iter().map(|p| bf16::from_f32(p / sum)).collect()
    };

    let w13_all: Vec<bf16> =
        (0..e * 2 * d_ff * d_model).map(|_| bf16::from_f32(rng.random_range(-0.05..0.05))).collect();
    let w2_all: Vec<bf16> = (0..e * d_ff * d_model).map(|_| bf16::from_f32(rng.random_range(-0.05..0.05))).collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff).map(|_| bf16::from_f32(rng.random_range(-0.01..0.01))).collect();
    let down_biases: Vec<bf16> = (0..e * d_model).map(|_| bf16::from_f32(rng.random_range(-0.01..0.01))).collect();

    let fused_kernel = MoeExpertsSingleDecodeKernels::<Metal>::new(ctx).expect("fused kernel");

    let x_buf = alloc_buffer_with_data(ctx, &x);
    let topk_ids_buf = alloc_buffer_with_data(ctx, &topk_ids);
    let topk_probs_buf = alloc_buffer_with_data(ctx, &topk_probs);
    let w13_buf = alloc_buffer_with_data(ctx, &w13_all);
    let w2_buf = alloc_buffer_with_data(ctx, &w2_all);
    let up_biases_buf = alloc_buffer_with_data(ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data(ctx, &down_biases);
    let hidden_buf = alloc_buffer::<f32>(ctx, k * d_ff);
    let y_buf = alloc_buffer::<bf16>(ctx, d_model);

    let make_args = || MoeExpertsSingleDecodeArguments {
        x: &x_buf,
        topk_ids: &topk_ids_buf,
        topk_probs: &topk_probs_buf,
        w13_all: &w13_buf,
        w2_all: &w2_buf,
        up_biases: &up_biases_buf,
        down_biases: &down_biases_buf,
        hidden: &hidden_buf,
        y: &y_buf,
        d_model,
        d_ff,
        k,
        gating_code: 2, // SwiGLU
        silu_alpha: 1.0,
        gate_clip_min: f32::NEG_INFINITY,
        gate_clip_max: f32::INFINITY,
        up_clip_min: f32::NEG_INFINITY,
        up_clip_max: f32::INFINITY,
        data_type: DataType::BF16,
    };

    for _ in 0..warmup {
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        fused_kernel.encode(&cb, make_args());
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        fused_kernel.encode(&cb, make_args());
        cb.commit();
        cb.wait_until_completed();
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
    eprintln!("    → Latency: {:.1} µs/token", mean * 1000.0);
}

#[test]
#[ignore]
fn test_fused_single_token_speed() {
    let ctx = create_ctx();

    let cases = vec![
        ("E16_D1024_F4096_K4", 1024, 4096, 16, 4, 3, 20),
        ("E8_D2048_F8192_K4", 2048, 8192, 8, 4, 3, 20),
        ("E8_D512_F2048_K2", 512, 2048, 8, 2, 3, 20),
    ];

    for (name, d_model, d_ff, e, k, warmup, iters) in cases {
        run_fused_single_token_case(&ctx, name, d_model, d_ff, e, k, warmup, iters);
    }
}

/// Compare indirect decode vs fused decode for single-token (T=1)
#[test]
#[ignore]
fn test_single_token_indirect_vs_fused() {
    let ctx = create_ctx();

    let cases = vec![
        ("E8_D512_F2048_K2", 512, 2048, 8, 2),
        ("E16_D1024_F4096_K4", 1024, 4096, 16, 4),
        ("E8_D2048_F8192_K4", 2048, 8192, 8, 4),
    ];

    eprintln!("\n=== Single-Token Decode: Indirect vs Fused ===\n");

    for (name, d_model, d_ff, e, k) in cases {
        let warmup = 3;
        let iters = 20;

        eprintln!("[{}] D={}, FF={}, E={}, K={}", name, d_model, d_ff, e, k);

        // Run indirect decode (T=1)
        let indirect_mean = run_indirect_decode_timed(&ctx, 1, d_model, d_ff, e, k, warmup, iters);

        // Run fused decode
        let fused_mean = run_fused_decode_timed(&ctx, d_model, d_ff, e, k, warmup, iters);

        let speedup = indirect_mean / fused_mean;
        eprintln!("    Indirect: {:.3}ms, Fused: {:.3}ms, Speedup: {:.2}x\n", indirect_mean, fused_mean, speedup);
    }
}

fn run_indirect_decode_timed(
    ctx: &MetalContext,
    t: usize,
    d_model: usize,
    d_ff: usize,
    e: usize,
    k: usize,
    warmup: usize,
    iters: usize,
) -> f64 {
    const K_TILE: usize = 64;

    let mut rng = StdRng::seed_from_u64(0xDEC0DE1234567890);
    let sum_k = t * k;

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

    let experts_kernel = MoeExpertsTwoPassDecodeBlock::<Metal>::new(ctx).expect("decode kernel");

    let x_perm_buf = alloc_buffer_with_data(ctx, &x_perm);
    let offsets_buf = alloc_buffer_with_data(ctx, &offsets);
    let w13_buf = alloc_buffer_with_data(ctx, &w13);
    let w2_buf = alloc_buffer_with_data(ctx, &w2);
    let up_biases_buf = alloc_buffer_with_data(ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data(ctx, &down_biases);

    let num_tiles_k = ((d_ff + K_TILE - 1) / K_TILE) as usize;
    let hidden_buf = alloc_buffer::<f32>(ctx, sum_k * d_ff);
    let output_buf = alloc_buffer::<bf16>(ctx, sum_k * d_model);

    let tile_counts_buf = alloc_buffer::<u32>(ctx, e);
    let tile_offsets_buf = alloc_buffer::<u32>(ctx, e + 1);
    let max_tiles = e * sum_k * 16384;
    let tile_map_buf = alloc_buffer::<u32>(ctx, max_tiles * 3);
    let total_tiles_buf = alloc_buffer::<u32>(ctx, 1);
    let dispatch_args_buf = alloc_buffer::<u32>(ctx, 3);
    let row_expert_map_buf = alloc_buffer::<u32>(ctx, sum_k);

    let make_args = || MoeExpertsTwoPassArguments {
        x_perm_buffer: &x_perm_buf,
        expert_offsets: &offsets_buf,
        row_expert_map: &row_expert_map_buf,
        hidden_buffer: &hidden_buf,
        output_buffer: &output_buf,
        w13_all: &w13_buf,
        w2_all: &w2_buf,
        up_biases: &up_biases_buf,
        down_biases: &down_biases_buf,
        tile_counts: &tile_counts_buf,
        tile_offsets: &tile_offsets_buf,
        tile_map: &tile_map_buf,
        total_tiles: &total_tiles_buf,
        dispatch_args: &dispatch_args_buf,
        total_rows: sum_k,
        d_model,
        d_ff,
        e,
        num_tiles_k: num_tiles_k as u32,
        gating_code: 2,
        gate_clip_min: f32::NEG_INFINITY,
        gate_clip_max: f32::INFINITY,
        up_clip_min: f32::NEG_INFINITY,
        up_clip_max: f32::INFINITY,
        silu_alpha: 1.0,
        data_type: DataType::BF16,
    };

    for _ in 0..warmup {
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        experts_kernel.encode(&cb, &make_args());
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        experts_kernel.encode(&cb, &make_args());
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    times.iter().sum::<f64>() / times.len() as f64
}

fn run_fused_decode_timed(
    ctx: &MetalContext,
    d_model: usize,
    d_ff: usize,
    e: usize,
    k: usize,
    warmup: usize,
    iters: usize,
) -> f64 {
    let mut rng = StdRng::seed_from_u64(0xF053ED);

    let x: Vec<bf16> = (0..d_model).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect();

    let topk_ids: Vec<i32> = (0..k).map(|i| (i % e) as i32).collect();

    let topk_probs: Vec<bf16> = {
        let raw: Vec<f32> = (0..k).map(|_| rng.random_range(0.1..1.0)).collect();
        let sum: f32 = raw.iter().sum();
        raw.iter().map(|p| bf16::from_f32(p / sum)).collect()
    };

    let w13_all: Vec<bf16> =
        (0..e * 2 * d_ff * d_model).map(|_| bf16::from_f32(rng.random_range(-0.05..0.05))).collect();
    let w2_all: Vec<bf16> = (0..e * d_ff * d_model).map(|_| bf16::from_f32(rng.random_range(-0.05..0.05))).collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff).map(|_| bf16::from_f32(rng.random_range(-0.01..0.01))).collect();
    let down_biases: Vec<bf16> = (0..e * d_model).map(|_| bf16::from_f32(rng.random_range(-0.01..0.01))).collect();

    let fused_kernel = MoeExpertsSingleDecodeKernels::<Metal>::new(ctx).expect("fused kernel");

    let x_buf = alloc_buffer_with_data(ctx, &x);
    let topk_ids_buf = alloc_buffer_with_data(ctx, &topk_ids);
    let topk_probs_buf = alloc_buffer_with_data(ctx, &topk_probs);
    let w13_buf = alloc_buffer_with_data(ctx, &w13_all);
    let w2_buf = alloc_buffer_with_data(ctx, &w2_all);
    let up_biases_buf = alloc_buffer_with_data(ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data(ctx, &down_biases);
    let hidden_buf = alloc_buffer::<f32>(ctx, k * d_ff);
    let y_buf = alloc_buffer::<bf16>(ctx, d_model);

    let make_args = || MoeExpertsSingleDecodeArguments {
        x: &x_buf,
        topk_ids: &topk_ids_buf,
        topk_probs: &topk_probs_buf,
        w13_all: &w13_buf,
        w2_all: &w2_buf,
        up_biases: &up_biases_buf,
        down_biases: &down_biases_buf,
        hidden: &hidden_buf,
        y: &y_buf,
        d_model,
        d_ff,
        k,
        gating_code: 2,
        silu_alpha: 1.0,
        gate_clip_min: f32::NEG_INFINITY,
        gate_clip_max: f32::INFINITY,
        up_clip_min: f32::NEG_INFINITY,
        up_clip_max: f32::INFINITY,
        data_type: DataType::BF16,
    };

    for _ in 0..warmup {
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        fused_kernel.encode(&cb, make_args());
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        fused_kernel.encode(&cb, make_args());
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    times.iter().sum::<f64>() / times.len() as f64
}
