use std::time::Instant;

use half::bf16;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    KernelDataType,
    kernel::moe::{
        MoeCountsOffsetsFusedArguments, MoeCountsOffsetsFusedKernel,
        MoeExpertsTwoPassArguments, MoeExpertsTwoPassDecodeKernel,
        MoeFinalizeArguments, MoeFinalizeKernel, MoeGatherArguments,
        MoeGatherKernel, MoeRouterTopKArguments, MoeRouterTopKKernel,
        MoeScatterKernels, MoeScatterWithMapArguments,
    },
};

use super::test_utils::{alloc_buffer, alloc_buffer_with_data, create_ctx};

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
        let x_buf = alloc_buffer_with_data(&ctx, &x);
        let router_w_buf = alloc_buffer_with_data(&ctx, &router_w);
        let router_b_buf = alloc_buffer_with_data(&ctx, &router_b);
        let topk_ids_buf = alloc_buffer::<i32>(&ctx, t * k);
        let topk_probs_buf = alloc_buffer::<bf16>(&ctx, t * k);

        let router_topk =
            MoeRouterTopKKernel::new(&ctx).expect("router+topk fused kernel");

        // Time fused Router+TopK
        let fused_perf = time_kernel("Router+TopK (FUSED)", 5, 20, || {
            let cb = ctx.command_queue.new_command_buffer();
            router_topk
                .encode(
                    &cb,
                    KernelDataType::BFloat16,
                    MoeRouterTopKArguments {
                        input_buffer: &x_buf,
                        weight_buffer: &router_w_buf,
                        bias_buffer: &router_b_buf,
                        topk_ids_buffer: &topk_ids_buf,
                        topk_probs_buffer: &topk_probs_buf,
                        t,
                        d_model,
                        e,
                        k,
                        renorm: true,
                    },
                )
                .expect("encode fused router+topk");
            cb.commit();
            cb.wait_until_completed();
        });

        fused_perf.print();
        eprintln!("    Total:   {:8.1} µs/token", fused_perf.mean_ms * 1000.0);
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
        let x_buf = alloc_buffer_with_data(&ctx, &x);
        let router_w_buf = alloc_buffer_with_data(&ctx, &router_w);
        let router_b_buf = alloc_buffer_with_data(&ctx, &router_b);
        let topk_ids_buf = alloc_buffer::<i32>(&ctx, t * k);
        let topk_probs_buf = alloc_buffer::<bf16>(&ctx, t * k);

        let router_topk =
            MoeRouterTopKKernel::new(&ctx).expect("router+topk fused kernel");

        // Time fused Router+TopK
        let fused_perf = time_kernel("Router+TopK (FUSED)", 5, 20, || {
            let cb = ctx.command_queue.new_command_buffer();
            router_topk
                .encode(
                    &cb,
                    KernelDataType::BFloat16,
                    MoeRouterTopKArguments {
                        input_buffer: &x_buf,
                        weight_buffer: &router_w_buf,
                        bias_buffer: &router_b_buf,
                        topk_ids_buffer: &topk_ids_buf,
                        topk_probs_buffer: &topk_probs_buf,
                        t,
                        d_model,
                        e,
                        k,
                        renorm: true,
                    },
                )
                .expect("encode fused router+topk");
            cb.commit();
            cb.wait_until_completed();
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

    let x_buf = alloc_buffer_with_data(&ctx, &x);
    let router_w_buf = alloc_buffer_with_data(&ctx, &router_w);
    let router_b_buf = alloc_buffer_with_data(&ctx, &router_b);
    let logits_buf = alloc_buffer::<bf16>(&ctx, t * e);
    let topk_ids_buf = alloc_buffer::<i32>(&ctx, t * k);
    let topk_probs_buf = alloc_buffer::<bf16>(&ctx, t * k);
    let num_blocks = ((t + 255) / 256).max(1);
    let num_tiles = ((e + 512 - 1) / 512).max(1);
    let partials_buf = alloc_buffer::<i32>(&ctx, num_blocks * num_tiles * e);
    let offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let sumk_buf = alloc_buffer::<u32>(&ctx, 1);
    let bucketed_ids_buf = alloc_buffer::<i32>(&ctx, t * k);
    let x_perm_buf = alloc_buffer::<bf16>(&ctx, t * k * d_model);
    let y_partial_buf = alloc_buffer::<bf16>(&ctx, t * k * d_model);
    let tok2row_buf = alloc_buffer::<i32>(&ctx, t * k);
    let y_out_buf = alloc_buffer::<bf16>(&ctx, t * d_model);

    // Expert weights buffers
    // Generate W13 in original layout [E, d_model, 2*d_ff]
    let w13_original: Vec<bf16> = (0..e * d_model * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
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
        .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.random_range(-0.1..0.1)))
        .collect();
    let down_biases: Vec<bf16> = (0..e * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.1..0.1)))
        .collect();

    let w13_buf = alloc_buffer_with_data(&ctx, &w13);
    let w2_buf = alloc_buffer_with_data(&ctx, &w2);
    let up_biases_buf = alloc_buffer_with_data(&ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data(&ctx, &down_biases);

    // Experts tiling buffers for two-pass
    const K_TILE: usize = 64;
    let sum_k = t * k;
    let num_tiles_k = (d_ff + K_TILE - 1) / K_TILE;
    let max_tiles = t * k * e * num_tiles_k;
    let tile_counts_buf = alloc_buffer::<u32>(&ctx, e);
    let tile_offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let tile_map_buf = alloc_buffer::<u32>(&ctx, max_tiles * 3);
    let total_tiles_buf = alloc_buffer::<u32>(&ctx, 1);
    let dispatch_args_buf = alloc_buffer::<u32>(&ctx, 3);

    // Two-pass specific buffers
    let hidden_buf = alloc_buffer::<f32>(&ctx, sum_k * d_ff);
    let row_expert_map_buf = alloc_buffer::<u32>(&ctx, sum_k);

    // Scatter block bases buffers
    let block_bases_buf = alloc_buffer::<u32>(&ctx, num_blocks * num_tiles);
    let block_alloc_buf = alloc_buffer::<u32>(&ctx, num_blocks * num_tiles);
    let bucketed_probs_buf = alloc_buffer::<bf16>(&ctx, t * k);

    // Create kernel structs (use production-validated encoding logic)
    let counts_offsets_kernel =
        MoeCountsOffsetsFusedKernel::new(&ctx).expect("counts+offsets fused");
    let scatter_kernel = MoeScatterKernels::new(&ctx).expect("scatter");
    let gather_kernel = MoeGatherKernel::new(&ctx).expect("gather");
    let experts_kernel = MoeExpertsTwoPassDecodeKernel::new(&ctx)
        .expect("experts two-pass decode");
    let finalize_kernel = MoeFinalizeKernel::new(&ctx).expect("finalize");
    let router_topk_fused_kernel =
        MoeRouterTopKKernel::new(&ctx).expect("router+topk fused");

    // Testing: Router + TopK + Counts+Offsets (FUSED)
    let router_topk_fused_perf =
        time_kernel("Router+TopK (FUSED)", 2, 5, || {
            let cb = ctx.command_queue.new_command_buffer();
            router_topk_fused_kernel
                .encode(
                    &cb,
                    KernelDataType::BFloat16,
                    MoeRouterTopKArguments {
                        input_buffer: &x_buf,
                        weight_buffer: &router_w_buf,
                        bias_buffer: &router_b_buf,
                        topk_ids_buffer: &topk_ids_buf,
                        topk_probs_buffer: &topk_probs_buf,
                        t,
                        d_model,
                        e,
                        k,
                        renorm: true,
                    },
                )
                .expect("router_topk_fused");
            cb.commit();
            cb.wait_until_completed();
        });

    let counts_offsets_perf =
        time_kernel("Counts+Offsets (FUSED)", 2, 5, || {
            let cb = ctx.command_queue.new_command_buffer();
            counts_offsets_kernel
                .encode(
                    &cb,
                    MoeCountsOffsetsFusedArguments {
                        topk_ids_buffer: &topk_ids_buf,
                        offsets_buffer: &offsets_buf,
                        sum_k_buffer: &sumk_buf,
                        partials_buffer: &partials_buf,
                        t,
                        e,
                        k,
                    },
                )
                .expect("counts+offsets fused");
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
                MoeExpertsTwoPassArguments {
                    x_perm_buffer: &x_perm_buf,
                    expert_offsets: &offsets_buf,
                    row_expert_map: &row_expert_map_buf,
                    hidden_buffer: &hidden_buf,
                    output_buffer: &y_partial_buf,
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

    eprintln!(
        "\n  ═══ Per-Kernel Latency (Production D=4096, H=14336, E=16, K=2, T=1) ═══"
    );
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
    eprintln!(
        "\n  Note: Times include Metal CB overhead (~10-50ms per kernel)."
    );
    eprintln!(
        "        Real GPU compute is much faster, but relative % shows bottleneck."
    );
}
