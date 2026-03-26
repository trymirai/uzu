#![cfg(metal_backend)]

use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::{
    DataType,
    backends::{
        common::{
            Backend, Encoder, Kernels,
            kernel::{
                MoeBlockBasesFromPartialsKernel, MoeCountsOffsetsFusedKernel, MoeFinalizeKernel, MoeRouterTopKKernel,
                MoeScatterBucketsMapKernel,
                moe::{MoeExpertsTwoPassArguments, MoeExpertsTwoPassDecodeBlock, MoeGatherArguments, MoeGatherKernels},
            },
        },
        metal::Metal,
    },
};

use super::moe_test_utils::{alloc_buffer, alloc_buffer_with_data, create_ctx};
use crate::common::perf::run_perf_with_warmup;

// Test E2E MoE performance with timing breakdown (decode mode, T=1)
#[test]
#[ignore]
fn test_moe_e2e_decode_perf() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xA0075);

    let configs =
        vec![("Small", 1, 1024, 256, 8, 2), ("Medium", 1, 2048, 1024, 16, 2), ("Production", 1, 4096, 14336, 16, 2)];

    eprintln!("\n=== End-to-End MoE Performance (DECODE, T=1) ===");

    for (config_name, t, d_model, d_ff, e, k) in configs {
        eprintln!("\n{}  T={}, D={}, H={}, E={}, K={}", config_name, t, d_model, d_ff, e, k);

        // Generate data
        let x: Vec<bf16> = (0..t * d_model).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect();
        let router_w: Vec<bf16> = (0..e * d_model).map(|_| bf16::from_f32(rng.random_range(-0.5..0.5))).collect();
        let router_b: Vec<bf16> = (0..e).map(|_| bf16::from_f32(rng.random_range(-0.1..0.1))).collect();

        // Buffers (simplified - just key buffers for timing)
        let x_buf = alloc_buffer_with_data(&ctx, &x);
        let router_w_buf = alloc_buffer_with_data(&ctx, &router_w);
        let router_b_buf = alloc_buffer_with_data(&ctx, &router_b);
        let mut topk_ids_buf = alloc_buffer::<i32>(&ctx, t * k);
        let mut topk_probs_buf = alloc_buffer::<bf16>(&ctx, t * k);

        let router_topk = <<Metal as Backend>::Kernels as Kernels>::MoeRouterTopKKernel::new(&ctx, DataType::BF16)
            .expect("router+topk fused kernel");

        // Time fused Router+TopK
        let fused_perf = run_perf_with_warmup("Router+TopK (FUSED)", 5, 20, || {
            let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
            router_topk.encode(
                &x_buf,
                &router_w_buf,
                &router_b_buf,
                &mut topk_ids_buf,
                &mut topk_probs_buf,
                t as u32,
                d_model as u32,
                e as u32,
                k as u32,
                true,
                &mut encoder,
            );
            encoder.end_encoding().submit().wait_until_completed().unwrap();
        });

        fused_perf.print();
        eprintln!("    Total:   {:8.1} µs/token", fused_perf.mean_ms * 1000.0);
    }
}

// Test E2E MoE performance with timing breakdown (prefill mode, T>1)
#[test]
#[ignore]
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
        eprintln!("\n{}  T={}, D={}, H={}, E={}, K={}", config_name, t, d_model, d_ff, e, k);

        // Generate data
        let x: Vec<bf16> = (0..t * d_model).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect();
        let router_w: Vec<bf16> = (0..e * d_model).map(|_| bf16::from_f32(rng.random_range(-0.5..0.5))).collect();
        let router_b: Vec<bf16> = (0..e).map(|_| bf16::from_f32(rng.random_range(-0.1..0.1))).collect();

        // Buffers (simplified - just key buffers for timing)
        let x_buf = alloc_buffer_with_data(&ctx, &x);
        let router_w_buf = alloc_buffer_with_data(&ctx, &router_w);
        let router_b_buf = alloc_buffer_with_data(&ctx, &router_b);
        let mut topk_ids_buf = alloc_buffer::<i32>(&ctx, t * k);
        let mut topk_probs_buf = alloc_buffer::<bf16>(&ctx, t * k);

        let router_topk = <<Metal as Backend>::Kernels as Kernels>::MoeRouterTopKKernel::new(&ctx, DataType::BF16)
            .expect("router+topk fused kernel");

        // Time fused Router+TopK
        let fused_perf = run_perf_with_warmup("Router+TopK (FUSED)", 5, 20, || {
            let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
            router_topk.encode(
                &x_buf,
                &router_w_buf,
                &router_b_buf,
                &mut topk_ids_buf,
                &mut topk_probs_buf,
                t as u32,
                d_model as u32,
                e as u32,
                k as u32,
                true,
                &mut encoder,
            );
            encoder.end_encoding().submit().wait_until_completed().unwrap();
        });

        fused_perf.print();
        eprintln!("    Total:   {:8.3} ms", fused_perf.mean_ms);
        eprintln!(
            "    Throughput: {:.1} tokens/sec, {:.3} ms/token",
            (t as f64 / fused_perf.mean_ms) * 1000.0,
            fused_perf.mean_ms / t as f64
        );
    }
}

// Test complete MoE pipeline timing breakdown (decode mode, T=1)
#[test]
#[ignore]
fn test_moe_pipeline_breakdown_decode() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xDECADE);

    eprintln!("\n=== MoE Pipeline Breakdown (DECODE, T=1) ===");
    eprintln!("Measures ALL MoE kernels: Router→TopK→Counts→Offsets→Scatter→Gather→Experts→Finalize\n");

    let (t, d_model, d_ff, e, k) = (1, 4096, 14336, 16, 2);

    // Allocate buffers
    let x: Vec<bf16> = (0..t * d_model).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect();
    let router_w: Vec<bf16> = (0..e * d_model).map(|_| bf16::from_f32(rng.random_range(-0.5..0.5))).collect();
    let router_b: Vec<bf16> = (0..e).map(|_| bf16::from_f32(rng.random_range(-0.1..0.1))).collect();

    let x_buf = alloc_buffer_with_data(&ctx, &x);
    let router_w_buf = alloc_buffer_with_data(&ctx, &router_w);
    let router_b_buf = alloc_buffer_with_data(&ctx, &router_b);
    let mut topk_ids_buf = alloc_buffer::<i32>(&ctx, t * k);
    let mut topk_probs_buf = alloc_buffer::<bf16>(&ctx, t * k);
    let num_blocks = ((t + 255) / 256).max(1);
    let num_tiles = ((e + 512 - 1) / 512).max(1);
    let mut partials_buf = alloc_buffer::<i32>(&ctx, num_blocks * num_tiles * e);
    let mut offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let mut sumk_buf = alloc_buffer::<u32>(&ctx, 1);
    let mut bucketed_ids_buf = alloc_buffer::<i32>(&ctx, t * k);
    let mut x_perm_buf = alloc_buffer::<bf16>(&ctx, t * k * d_model);
    let mut y_partial_buf = alloc_buffer::<bf16>(&ctx, t * k * d_model);
    let mut tok2row_buf = alloc_buffer::<i32>(&ctx, t * k);
    let mut y_out_buf = alloc_buffer::<bf16>(&ctx, t * d_model);

    // Expert weights buffers
    // Generate W13 in original layout [E, d_model, 2*d_ff]
    let w13_original: Vec<bf16> =
        (0..e * d_model * 2 * d_ff).map(|_| bf16::from_f32(rng.random_range(-0.5..0.5))).collect();

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

    let w2: Vec<bf16> = (0..e * d_ff * d_model).map(|_| bf16::from_f32(rng.random_range(-0.5..0.5))).collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff).map(|_| bf16::from_f32(rng.random_range(-0.1..0.1))).collect();
    let down_biases: Vec<bf16> = (0..e * d_model).map(|_| bf16::from_f32(rng.random_range(-0.1..0.1))).collect();

    let w13_buf = alloc_buffer_with_data(&ctx, &w13);
    let w2_buf = alloc_buffer_with_data(&ctx, &w2);
    let up_biases_buf = alloc_buffer_with_data(&ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data(&ctx, &down_biases);

    // Experts tiling buffers for two-pass
    const K_TILE: usize = 64;
    let sum_k = t * k;
    let num_tiles_k = (d_ff + K_TILE - 1) / K_TILE;
    let max_tiles = t * k * e * num_tiles_k;
    let mut tile_counts_buf = alloc_buffer::<u32>(&ctx, e);
    let mut tile_offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let mut tile_map_buf = alloc_buffer::<u32>(&ctx, max_tiles * 3);
    let mut total_tiles_buf = alloc_buffer::<u32>(&ctx, 1);
    let mut dispatch_args_buf = alloc_buffer::<u32>(&ctx, 3);

    // Two-pass specific buffers
    let mut hidden_buf = alloc_buffer::<f32>(&ctx, sum_k * d_ff);
    let mut row_expert_map_buf = alloc_buffer::<u32>(&ctx, sum_k);

    // Scatter block bases buffers
    let mut block_bases_buf = alloc_buffer::<u32>(&ctx, num_blocks * num_tiles);
    let mut block_alloc_buf = alloc_buffer::<u32>(&ctx, num_blocks * num_tiles);
    let mut bucketed_probs_buf = alloc_buffer::<bf16>(&ctx, t * k);

    // Create kernel structs (use production-validated encoding logic)
    let counts_offsets_kernel =
        <<Metal as Backend>::Kernels as Kernels>::MoeCountsOffsetsFusedKernel::new(&ctx).expect("counts+offsets fused");
    let scatter_bases_kernel = <<Metal as Backend>::Kernels as Kernels>::MoeBlockBasesFromPartialsKernel::new(&ctx)
        .expect("<<Metal as Backend>::Kernels as Kernels>::MoeBlockBasesFromPartialsKernel");
    let scatter_map_kernel =
        <<Metal as Backend>::Kernels as Kernels>::MoeScatterBucketsMapKernel::new(&ctx, DataType::BF16)
            .expect("<<Metal as Backend>::Kernels as Kernels>::MoeScatterBucketsMapKernel");
    let gather_kernel = MoeGatherKernels::<Metal>::new(&ctx).expect("gather");
    let experts_kernel = MoeExpertsTwoPassDecodeBlock::<Metal>::new(&ctx).expect("experts two-pass decode");
    let finalize_kernel =
        <<Metal as Backend>::Kernels as Kernels>::MoeFinalizeKernel::new(&ctx, DataType::BF16).expect("finalize");
    let router_topk_fused_kernel =
        <<Metal as Backend>::Kernels as Kernels>::MoeRouterTopKKernel::new(&ctx, DataType::BF16)
            .expect("router+topk fused");

    // Testing: Router + TopK + Counts+Offsets (FUSED)
    let router_topk_fused_perf = run_perf_with_warmup("Router+TopK (FUSED)", 2, 5, || {
        let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
        router_topk_fused_kernel.encode(
            &x_buf,
            &router_w_buf,
            &router_b_buf,
            &mut topk_ids_buf,
            &mut topk_probs_buf,
            t as u32,
            d_model as u32,
            e as u32,
            k as u32,
            true,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();
    });

    let counts_offsets_perf = run_perf_with_warmup("Counts+Offsets (FUSED)", 2, 5, || {
        let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
        counts_offsets_kernel.encode(
            &topk_ids_buf,
            &mut offsets_buf,
            &mut sumk_buf,
            &mut partials_buf,
            t as u32,
            e as u32,
            k as u32,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();
    });

    let scatter_perf = run_perf_with_warmup("Scatter", 2, 5, || {
        let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
        scatter_bases_kernel.encode(
            &partials_buf,
            &mut block_bases_buf,
            &mut block_alloc_buf,
            e as u32,
            num_blocks as u32,
            num_tiles as u32,
            0u32,
            &mut encoder,
        );
        scatter_map_kernel.encode(
            &topk_ids_buf,
            &topk_probs_buf,
            &offsets_buf,
            &block_bases_buf,
            &block_alloc_buf,
            &mut bucketed_ids_buf,
            &mut bucketed_probs_buf,
            t as u32,
            e as u32,
            k as u32,
            num_blocks as u32,
            num_tiles as u32,
            &mut tok2row_buf,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();
    });

    let gather_perf = run_perf_with_warmup("Gather", 2, 5, || {
        let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
        gather_kernel.encode(
            &mut encoder,
            DataType::BF16,
            MoeGatherArguments {
                x_buffer: &x_buf,
                bucketed_ids_buffer: &bucketed_ids_buf,
                x_perm_buffer: &mut x_perm_buf,
                sumk_buffer: &sumk_buf,
                t,
                k,
                d_model,
            },
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();
    });

    let experts_perf = run_perf_with_warmup("Experts (MAIN COMPUTE)", 2, 5, || {
        let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
        experts_kernel.encode(
            &mut encoder,
            MoeExpertsTwoPassArguments {
                x_perm_buffer: &x_perm_buf,
                expert_offsets: &offsets_buf,
                row_expert_map: &mut row_expert_map_buf,
                hidden_buffer: &mut hidden_buf,
                output_buffer: &mut y_partial_buf,
                w13_all: &w13_buf,
                w2_all: &w2_buf,
                up_biases: &up_biases_buf,
                down_biases: &down_biases_buf,
                tile_counts: &mut tile_counts_buf,
                tile_offsets: &mut tile_offsets_buf,
                tile_map: &mut tile_map_buf,
                total_tiles: &mut total_tiles_buf,
                dispatch_args: &mut dispatch_args_buf,
                total_rows: sum_k,
                d_model,
                d_ff,
                e,
                num_tiles_k: num_tiles_k as u32,
                gating_code: 2, // SILU
                gate_clip_min: f32::NEG_INFINITY,
                gate_clip_max: 20.0,
                up_clip_min: -19.0,
                up_clip_max: 21.0,
                silu_alpha: 1.702,
                data_type: DataType::BF16,
            },
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();
    });

    let finalize_perf = run_perf_with_warmup("Finalize", 2, 5, || {
        let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
        finalize_kernel.encode(
            &tok2row_buf,
            &topk_probs_buf,
            &y_partial_buf,
            &mut y_out_buf,
            t as u32,
            d_model as u32,
            k as u32,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();
    });

    // Print results
    router_topk_fused_perf.print();
    counts_offsets_perf.print();
    scatter_perf.print();
    gather_perf.print();
    experts_perf.print();
    finalize_perf.print();

    // Calculate breakdown
    let total_us = (router_topk_fused_perf.mean_ms
        + counts_offsets_perf.mean_ms
        + scatter_perf.mean_ms
        + gather_perf.mean_ms
        + experts_perf.mean_ms
        + finalize_perf.mean_ms)
        * 1000.0;

    eprintln!("\n  ═══ Per-Kernel Latency (Production D=4096, H=14336, E=16, K=2, T=1) ═══");
    eprintln!(
        "    Router+TopK (FUSED): {:8.1} us  ({:5.1}%)",
        router_topk_fused_perf.mean_ms * 1000.0,
        (router_topk_fused_perf.mean_ms / (total_us / 1000.0)) * 100.0
    );
    eprintln!(
        "    Counts+Offsets: {:8.1} us  ({:5.1}%)",
        counts_offsets_perf.mean_ms * 1000.0,
        (counts_offsets_perf.mean_ms / (total_us / 1000.0)) * 100.0
    );
    eprintln!(
        "    Scatter:     {:8.1} us  ({:5.1}%)",
        scatter_perf.mean_ms * 1000.0,
        (scatter_perf.mean_ms / (total_us / 1000.0)) * 100.0
    );
    eprintln!(
        "    Gather:      {:8.1} us  ({:5.1}%)",
        gather_perf.mean_ms * 1000.0,
        (gather_perf.mean_ms / (total_us / 1000.0)) * 100.0
    );
    eprintln!(
        "    Experts:     {:8.1} us  ({:5.1}%) ← MAIN COMPUTE",
        experts_perf.mean_ms * 1000.0,
        (experts_perf.mean_ms / (total_us / 1000.0)) * 100.0
    );
    eprintln!(
        "    Finalize:    {:8.1} us  ({:5.1}%)",
        finalize_perf.mean_ms * 1000.0,
        (finalize_perf.mean_ms / (total_us / 1000.0)) * 100.0
    );
    eprintln!("    ═══════════════════════════════════════════");
    eprintln!("    TOTAL:       {:8.1} us (100.0%)", total_us);
    eprintln!("\n  Note: Times include Metal CB overhead (~10-50ms per kernel).");
    eprintln!("        Real GPU compute is much faster, but relative % shows bottleneck.");
}
