use std::time::Instant;

use half::bf16;
use metal::MTLResourceOptions;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::moe::{
        MoeExpertsArguments, MoeExpertsFusedKernel, MoeExpertsTwoPassArguments,
        MoeExpertsTwoPassDecodeKernel, MoeExpertsTwoPassPrefillKernel,
    },
};

use super::test_utils::create_ctx;

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

    let x_perm_buf = ctx.device.new_buffer_with_data(
        x_perm.as_ptr() as *const _,
        (x_perm.len() * size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let offsets_buf = ctx.device.new_buffer_with_data(
        offsets.as_ptr() as *const _,
        (offsets.len() * size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let w13_buf = ctx.device.new_buffer_with_data(
        w13.as_ptr() as *const _,
        (w13.len() * size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let w2_buf = ctx.device.new_buffer_with_data(
        w2.as_ptr() as *const _,
        (w2.len() * size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let up_biases_buf = ctx.device.new_buffer_with_data(
        up_biases.as_ptr() as *const _,
        (up_biases.len() * size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let down_biases_buf = ctx.device.new_buffer_with_data(
        down_biases.as_ptr() as *const _,
        (down_biases.len() * size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let num_tiles_k = ((d_ff + K_TILE - 1) / K_TILE) as usize;

    let hidden_buf = ctx.device.new_buffer(
        (sum_k * d_ff * size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let partial_buf = ctx.device.new_buffer(
        (num_tiles_k * sum_k * d_model * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = ctx.device.new_buffer(
        (sum_k * d_model * size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Buffers for indirect dispatch
    let tile_counts_buf = ctx.device.new_buffer(
        (e * size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let tile_offsets_buf = ctx.device.new_buffer(
        ((e + 1) * size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let max_tiles = e * sum_k * 16384; // Conservative upper bound
    let tile_map_buf = ctx.device.new_buffer(
        (max_tiles * 3 * size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let total_tiles_buf = ctx.device.new_buffer(
        size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let dispatch_args_buf = ctx.device.new_buffer(
        (3 * size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let row_expert_map_buf = ctx.device.new_buffer(
        (sum_k * size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

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
fn run_prefill_case(
    ctx: &MTLContext,
    mode: &str,
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

    eprintln!(
        "\n[{}] {} => T={}, D={}, FF={}, E={}, K={}, sum_k={}",
        mode, name, t, d_model, d_ff, e, k, sum_k
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

    let x_perm_buf = ctx.device.new_buffer_with_data(
        x_perm.as_ptr() as *const _,
        (x_perm.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let offsets_buf = ctx.device.new_buffer_with_data(
        offsets.as_ptr() as *const _,
        (offsets.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
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

    let y_partial_buf = ctx.device.new_buffer(
        (sum_k * d_model * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    const BN: usize = 64;
    let num_tiles_n = (d_model + BN - 1) / BN;
    let max_tiles = sum_k * e * num_tiles_n;
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

    let experts_kernel =
        MoeExpertsFusedKernel::new(ctx).expect("experts kernel");

    let make_args = || MoeExpertsArguments {
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
        experts_kernel.encode(&cb, make_args()).expect("encode");
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let cb = ctx.command_queue.new_command_buffer();
        experts_kernel.encode(&cb, make_args()).expect("encode");
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

#[test]
fn test_gemv_prefill_speed() {
    let ctx = create_ctx();

    let cases = vec![
        ("T32_E16_D1024_F4096", 32, 1024, 4096, 16, 4, 2, 10),
        ("T64_E16_D2048_F6144", 64, 2048, 6144, 16, 4, 2, 10),
    ];

    for (name, t, d_model, d_ff, e, k, warmup, iters) in cases {
        run_prefill_case(
            &ctx, "prefill", name, t, d_model, d_ff, e, k, warmup, iters,
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
    use std::mem::size_of;

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

    let x_perm_buf = ctx.device.new_buffer_with_data(
        x_perm.as_ptr() as *const _,
        (x_perm.len() * size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let offsets_buf = ctx.device.new_buffer_with_data(
        offsets.as_ptr() as *const _,
        (offsets.len() * size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w13_buf = ctx.device.new_buffer_with_data(
        w13.as_ptr() as *const _,
        (w13.len() * size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w2_buf = ctx.device.new_buffer_with_data(
        w2.as_ptr() as *const _,
        (w2.len() * size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let up_biases_buf = ctx.device.new_buffer_with_data(
        up_biases.as_ptr() as *const _,
        (up_biases.len() * size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let down_biases_buf = ctx.device.new_buffer_with_data(
        down_biases.as_ptr() as *const _,
        (down_biases.len() * size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let num_tiles_k = ((d_ff + K_TILE - 1) / K_TILE) as usize;

    let hidden_buf = ctx.device.new_buffer(
        (sum_k * d_ff * size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let partial_buf = ctx.device.new_buffer(
        (num_tiles_k * sum_k * d_model * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = ctx.device.new_buffer(
        (sum_k * d_model * size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let tile_counts_buf = ctx.device.new_buffer(
        (e * size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let tile_offsets_buf = ctx.device.new_buffer(
        ((e + 1) * size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let max_tiles = e * sum_k * 16384;
    let tile_map_buf = ctx.device.new_buffer(
        (max_tiles * 3 * size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let total_tiles_buf = ctx.device.new_buffer(
        size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let dispatch_args_buf = ctx.device.new_buffer(
        (3 * size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let row_expert_map_buf = ctx.device.new_buffer(
        (sum_k * size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

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
