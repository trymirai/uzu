// Quick GEMV performance test for T=1 decode
use std::time::Instant;

use half::bf16;
use metal::{Device, MTLResourceOptions};
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::moe::{MoeExpertsArguments, MoeExpertsKernel},
};

fn create_ctx() -> MTLContext {
    let device = metal::Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    MTLContext::new(device, queue).expect("ctx")
}

#[test]
fn test_gemv_decode_speed() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xDECADE);

    eprintln!("\n=== GEMV Decode Performance (T=1, K=2) ===");

    let (t, d_model, d_ff, e, k) = (1, 4096, 14336, 16, 2);
    let sum_k = t * k;

    eprintln!(
        "Config: T={}, D={}, FF={}, E={}, K={}, sum_k={}",
        t, d_model, d_ff, e, k, sum_k
    );

    // Allocate minimal buffers for Experts kernel
    let x_perm: Vec<bf16> = (0..sum_k * d_model)
        .map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0)))
        .collect();
    let offsets: Vec<u32> =
        (0..=e).map(|i| i as u32 * (sum_k / e) as u32).collect();
    let w13: Vec<bf16> = (0..e * d_model * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.gen_range(-0.1..0.1)))
        .collect();
    let w2: Vec<bf16> = (0..e * d_ff * d_model)
        .map(|_| bf16::from_f32(rng.gen_range(-0.1..0.1)))
        .collect();
    let up_biases: Vec<bf16> =
        (0..e * 2 * d_ff).map(|_| bf16::from_f32(0.0)).collect();
    let down_biases: Vec<bf16> =
        (0..e * d_model).map(|_| bf16::from_f32(0.0)).collect();

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

    // Tiling buffers (not used by GEMV but required by API)
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

    // Create kernel
    let experts_kernel = MoeExpertsKernel::new(&ctx).expect("experts kernel");

    eprintln!("Warming up (1 iter)...");
    for _ in 0..1 {
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
                    gating_code: 2, // SwiGLU
                    gate_clip_min: f32::NEG_INFINITY,
                    gate_clip_max: f32::INFINITY,
                    up_clip_min: f32::NEG_INFINITY,
                    up_clip_max: f32::INFINITY,
                    silu_alpha: 1.0,
                    data_type: KernelDataType::BFloat16,
                },
            )
            .expect("encode");
        cb.commit();
        cb.wait_until_completed();
    }

    eprintln!("Measuring (10 iters)...");
    let iters = 10;
    let start = Instant::now();
    for _ in 0..iters {
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
                    gating_code: 2, // SwiGLU
                    gate_clip_min: f32::NEG_INFINITY,
                    gate_clip_max: f32::INFINITY,
                    up_clip_min: f32::NEG_INFINITY,
                    up_clip_max: f32::INFINITY,
                    silu_alpha: 1.0,
                    data_type: KernelDataType::BFloat16,
                },
            )
            .expect("encode");
        cb.commit();
        cb.wait_until_completed();
    }
    let elapsed = start.elapsed();
    let mean_ms = elapsed.as_micros() as f64 / 1000.0 / iters as f64;

    eprintln!("\n✓ GEMV Experts (T=1): {:.3} ms/iter", mean_ms);
    eprintln!("  (This uses the decode-specialized GEMV path for T=1)\n");
}
