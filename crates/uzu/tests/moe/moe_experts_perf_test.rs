use std::time::Instant;

use half::bf16;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::moe::{
        MoeExpertsTwoPassArguments, MoeExpertsTwoPassDecodeKernel,
        MoeExpertsTwoPassPrefillKernel,
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
    ctx: &MTLContext,
    name: &str,
    t: usize,
    d_model: usize,
    d_ff: usize,
    e: usize,
    k: usize,
    warmup: usize,
    iters: usize,
) {
    use std::mem::size_of;

    const K_TILE: usize = 64;

    let mut rng = StdRng::seed_from_u64(0xDEC0DE1234567890);
    let sum_k = t * k;

    eprintln!(
        "\n[decode/two-pass] {} => T={}, D={}, FF={}, E={}, K={}, sum_k={}",
        name, t, d_model, d_ff, e, k, sum_k
    );

    let x_perm: Vec<bf16> = (0..sum_k * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
        .collect();
    let offsets = build_offsets(e, sum_k);

    // Generate W13 in original layout [E, d_model, 2*d_ff]
    let w13_original: Vec<bf16> = (0..e * d_model * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.random_range(-0.05..0.05)))
        .collect();

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
    let w2: Vec<bf16> = (0..e * d_ff * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.05..0.05)))
        .collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.random_range(-0.01..0.01)))
        .collect();
    let down_biases: Vec<bf16> = (0..e * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.01..0.01)))
        .collect();

    let experts_kernel =
        MoeExpertsTwoPassDecodeKernel::new(ctx).expect("experts decode kernel");

    let x_perm_buf = alloc_buffer_with_data(&ctx, &x_perm);
    let offsets_buf = alloc_buffer_with_data(&ctx, &offsets);

    let w13_buf = alloc_buffer_with_data(&ctx, &w13);

    let w2_buf = alloc_buffer_with_data(&ctx, &w2);

    let up_biases_buf = alloc_buffer_with_data(&ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data(&ctx, &down_biases);

    let num_tiles_k = ((d_ff + K_TILE - 1) / K_TILE) as usize;

    let hidden_buf = alloc_buffer::<f32>(&ctx, sum_k * d_ff);
    let partial_buf = alloc_buffer::<f32>(&ctx, num_tiles_k * sum_k * d_model);
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
        partial_buffer: &partial_buf,
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
        data_type: KernelDataType::BFloat16,
    };

    for _ in 0..warmup {
        let cb = ctx.command_queue.new_command_buffer();
        experts_kernel
            .encode(&cb, make_two_pass_args())
            .expect("two-pass encode");
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let cb = ctx.command_queue.new_command_buffer();
        experts_kernel
            .encode(&cb, make_two_pass_args())
            .expect("two-pass encode");
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    let var = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
        / times.len() as f64;
    let std = var.sqrt();

    eprintln!(
        "    mean={:.3}ms median={:.3}ms min={:.3}ms max={:.3}ms std={:.3}ms",
        mean, median, min, max, std
    );
    eprintln!("    → Latency: {:.1} µs/token", (mean / t as f64) * 1000.0);
}

#[test]
fn test_two_pass_decode_speed() {
    let ctx = create_ctx();

    let cases = vec![
        ("T1_E16_D1024_F4096", 1, 1024, 4096, 16, 4, 3, 20),
        ("T1_E8_D2048_F8192", 1, 2048, 8192, 8, 4, 3, 20),
    ];

    for (name, t, d_model, d_ff, e, k, warmup, iters) in &cases {
        run_decode_case(
            &ctx, name, *t, *d_model, *d_ff, *e, *k, *warmup, *iters,
        );
    }
}

fn run_two_pass_prefill_case(
    ctx: &MTLContext,
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

    eprintln!(
        "\n[2-pass prefill] {} => T={}, D={}, FF={}, E={}, K={}, sum_k={}",
        name, t, d_model, d_ff, e, k, sum_k
    );

    let x_perm: Vec<bf16> = (0..sum_k * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
        .collect();
    let offsets = build_offsets(e, sum_k);

    let w13_original: Vec<bf16> = (0..e * d_model * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.random_range(-0.05..0.05)))
        .collect();

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
    let w2: Vec<bf16> = (0..e * d_ff * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.05..0.05)))
        .collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.random_range(-0.01..0.01)))
        .collect();
    let down_biases: Vec<bf16> = (0..e * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.01..0.01)))
        .collect();

    let experts_kernel = MoeExpertsTwoPassPrefillKernel::new(ctx)
        .expect("experts prefill kernel");

    let x_perm_buf = alloc_buffer_with_data(&ctx, &x_perm);
    let offsets_buf = alloc_buffer_with_data(&ctx, &offsets);
    let w13_buf = alloc_buffer_with_data(&ctx, &w13);
    let w2_buf = alloc_buffer_with_data(&ctx, &w2);
    let up_biases_buf = alloc_buffer_with_data(&ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data(&ctx, &down_biases);

    let num_tiles_k = ((d_ff + K_TILE - 1) / K_TILE) as usize;

    let hidden_buf = alloc_buffer::<f32>(&ctx, sum_k * d_ff);
    let partial_buf = alloc_buffer::<f32>(&ctx, num_tiles_k * sum_k * d_model);
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
        partial_buffer: &partial_buf,
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
        data_type: KernelDataType::BFloat16,
    };

    for _ in 0..warmup {
        let cb = ctx.command_queue.new_command_buffer();
        experts_kernel
            .encode(&cb, make_two_pass_args())
            .expect("two-pass prefill encode");
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let cb = ctx.command_queue.new_command_buffer();
        experts_kernel
            .encode(&cb, make_two_pass_args())
            .expect("two-pass prefill encode");
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    let var = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
        / times.len() as f64;
    let std = var.sqrt();

    eprintln!(
        "    mean={:.3}ms median={:.3}ms min={:.3}ms max={:.3}ms std={:.3}ms",
        mean, median, min, max, std
    );
    eprintln!(
        "    → Throughput: {:.1} µs/token (mean / sum_k)",
        (mean / sum_k as f64) * 1000.0
    );
}

#[test]
fn test_two_pass_prefill_speed() {
    let ctx = create_ctx();

    let cases = vec![
        ("T32_E16_D1024_F4096", 32, 1024, 4096, 16, 4, 2, 10),
        ("T64_E16_D2048_F6144", 64, 2048, 6144, 16, 4, 2, 10),
    ];

    for (name, t, d_model, d_ff, e, k, warmup, iters) in cases {
        run_two_pass_prefill_case(
            &ctx, name, t, d_model, d_ff, e, k, warmup, iters,
        );
    }
}
