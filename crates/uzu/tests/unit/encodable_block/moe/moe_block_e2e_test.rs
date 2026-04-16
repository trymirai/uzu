use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::{
    DataType,
    backends::common::{
        Backend, Encoder, Kernels,
        gpu_types::{ActivationType, activation_silu_alpha},
        kernel::{
            MoeBlockBasesFromPartialsKernel, MoeCountsOffsetsFusedKernel, MoeFinalizeKernel, MoeRouterTopKKernel,
            MoeScatterBucketsMapKernel,
            moe::{MoeExpertsTwoPassArguments, MoeExpertsTwoPassPrefillBlock, MoeGatherArguments, MoeGatherKernels},
        },
    },
};

use crate::encodable_block::mlp::moe::tests::common::helpers::{
    alloc_allocation, alloc_allocation_with_data, alloc_buffer_with_data, allocation_prefix_to_vec, allocation_to_vec,
    create_context,
};

fn moe_cpu_reference(
    x: &[bf16],
    router_weight: &[f32], // [E, d_model] - kept as F32 for router (computed before BF16 conversion)
    router_bias: &[f32],   // [E]
    w13: &[bf16],          // source layout [E, d_model, 2*d_ff] (GPU transposes to [E, 2*d_ff, d_model])
    w2: &[bf16],           // [E, d_ff, d_model]
    up_biases: &[bf16],    // [E, 2*d_ff]
    down_biases: &[bf16],  // [E, d_model]
    t: usize,
    e: usize,
    k: usize,
    d_model: usize,
    d_ff: usize,
    gating_code: u32, // 0=GELU, 1=SiLU, 2=SwiGLU, 3=GEGLU
    silu_alpha: f32,
    gate_clip: (f32, f32),
    up_clip: (f32, f32),
) -> Vec<f32> {
    let mut output = vec![0.0f32; t * d_model];

    for token_idx in 0..t {
        let x_start = token_idx * d_model;

        // Router: compute logits = x @ W_r^T + b_r
        // Router uses F32 precision (router_weight/router_bias are F32)
        let mut token_logits = vec![0.0f32; e];
        for expert_idx in 0..e {
            let mut logit = router_bias[expert_idx];
            for d in 0..d_model {
                let x_val = f32::from(x[x_start + d]);
                logit += x_val * router_weight[expert_idx * d_model + d];
            }
            token_logits[expert_idx] = logit;
        }

        // TopK selection
        let mut indices_and_logits: Vec<(usize, f32)> = token_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indices_and_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_k_indices: Vec<usize> = indices_and_logits[..k].iter().map(|&(i, _)| i).collect();
        let top_k_logits: Vec<f32> = indices_and_logits[..k].iter().map(|&(_, v)| v).collect();

        // Softmax normalization (matching Python: jax.nn.softmax)
        let max_logit = top_k_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = top_k_logits.iter().map(|&v| (v - max_logit).exp()).sum();
        let weights: Vec<f32> = top_k_logits.iter().map(|&v| (v - max_logit).exp() / exp_sum).collect();

        // Process each selected expert (matching Python: vmap(run_expert))
        for (expert_idx, &weight) in top_k_indices.iter().zip(weights.iter()) {
            let x_start = token_idx * d_model;

            // Expert MLP weights layout: CPU uses W13[E, d_model, 2*d_ff], GPU stores as [E, 2*d_ff, d_model]; W2[E, d_ff, d_model]
            let w13_offset = expert_idx * d_model * 2 * d_ff;
            let w2_offset = expert_idx * d_ff * d_model;
            let bias_up_offset = expert_idx * 2 * d_ff;
            let bias_down_offset = expert_idx * d_model;

            // Up projection: W13[expert, input, output] where output=[up, gate]
            // Convert BF16→F32 on each access to match GPU precision
            let mut up_out = vec![0.0f32; d_ff];
            let mut gate_out = vec![0.0f32; d_ff];

            for ff_idx in 0..d_ff {
                let mut up_sum = f32::from(up_biases[bias_up_offset + ff_idx]);
                let mut gate_sum = f32::from(up_biases[bias_up_offset + d_ff + ff_idx]);

                for input_idx in 0..d_model {
                    let w_base = w13_offset + input_idx * 2 * d_ff;
                    let x_val = f32::from(x[x_start + input_idx]);
                    let w_up = f32::from(w13[w_base + ff_idx]);
                    let w_gate = f32::from(w13[w_base + d_ff + ff_idx]);
                    up_sum += x_val * w_up;
                    gate_sum += x_val * w_gate;
                }

                up_out[ff_idx] = up_sum;
                gate_out[ff_idx] = gate_sum;
            }

            // Apply clipping
            for i in 0..d_ff {
                gate_out[i] = gate_out[i].clamp(gate_clip.0, gate_clip.1);
                up_out[i] = up_out[i].clamp(up_clip.0, up_clip.1);
            }

            // Activation: depends on gating_code
            // 0=GELU(up), 1=SiLU(up), 2=SwiGLU(gate)*up, 3=GEGLU(gate)*up
            let mut hidden = vec![0.0f32; d_ff];
            for i in 0..d_ff {
                hidden[i] = match gating_code {
                    0 => ActivationType::GELU.activate(up_out[i]),
                    1 => activation_silu_alpha(up_out[i], silu_alpha),
                    2 => activation_silu_alpha(gate_out[i], silu_alpha) * up_out[i],
                    3 => ActivationType::GELU.activate(gate_out[i]) * up_out[i],
                    _ => activation_silu_alpha(gate_out[i], silu_alpha) * up_out[i], // fallback to SwiGLU
                };
            }

            // Down projection: W2[expert, ff, output]
            // Convert BF16→F32 on each access to match GPU precision
            // CRITICAL: Two-pass kernel accumulates in F32, only quantizes at the very end
            for out_idx in 0..d_model {
                let mut fc2_sum = 0.0f32;
                for ff_idx in 0..d_ff {
                    let w2_val = f32::from(w2[w2_offset + ff_idx * d_model + out_idx]);
                    fc2_sum += hidden[ff_idx] * w2_val;
                }
                // Two-pass: keep in F32 (no intermediate quantization)

                // Add down bias in F32
                let down_bias_val = f32::from(down_biases[bias_down_offset + out_idx]);
                let with_bias = fc2_sum + down_bias_val;

                // Only quantize to BF16 once at the end (matches two-pass final write)
                let with_bias_bf16 = bf16::from_f32(with_bias);
                let final_val = f32::from(with_bias_bf16);

                output[x_start + out_idx] += weight * final_val;
            }
        }
    }

    output
}

// Main entry point - automatically tests both modes for T>1
fn run_moe_parity_test<B: Backend>(
    ctx: &B::Context,
    t: usize,
    e: usize,
    k: usize,
    d_model: usize,
    d_ff: usize,
    gating_code: u32,
    silu_alpha: f32,
    gate_clip: (f32, f32),
    up_clip: (f32, f32),
    seed: u64,
    test_name: &str,
) {
    if t > 1 {
        // Test 1-pass prefill (default, stable)
        run_moe_parity_test_internal::<B>(
            ctx,
            t,
            e,
            k,
            d_model,
            d_ff,
            gating_code,
            silu_alpha,
            gate_clip,
            up_clip,
            seed,
            &format!("{}_decode", test_name),
        );
        // Test 2-pass prefill (experimental)
        run_moe_parity_test_internal::<B>(
            ctx,
            t,
            e,
            k,
            d_model,
            d_ff,
            gating_code,
            silu_alpha,
            gate_clip,
            up_clip,
            seed,
            &format!("{}_prefill", test_name),
        );
    } else {
        // Decode mode (T=1) - no prefill variants
        run_moe_parity_test_internal::<B>(
            ctx,
            t,
            e,
            k,
            d_model,
            d_ff,
            gating_code,
            silu_alpha,
            gate_clip,
            up_clip,
            seed,
            test_name,
        );
    }
}

fn run_moe_parity_test_internal<B: Backend>(
    ctx: &B::Context,
    t: usize,
    e: usize,
    k: usize,
    d_model: usize,
    d_ff: usize,
    gating_code: u32, // 0=GELU, 1=SiLU, 2=SwiGLU, 3=GEGLU
    silu_alpha: f32,
    gate_clip: (f32, f32),
    up_clip: (f32, f32),
    seed: u64,
    test_name: &str,
) {
    let mut rng = StdRng::seed_from_u64(seed);

    let prefill_mode = if t > 1 {
        "2-pass"
    } else {
        "decode"
    };

    eprintln!(
        "\n[{}] T={}, E={}, K={}, d_model={}, d_ff={}, alpha={}, gate_clip={:?}, up_clip={:?}, mode={}",
        test_name, t, e, k, d_model, d_ff, silu_alpha, gate_clip, up_clip, prefill_mode
    );

    // Random BF16 inputs
    let x: Vec<bf16> = (0..t * d_model).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect();

    // Generate random router weights and biases for CPU reference
    let router_weight_f32: Vec<f32> = (0..e * d_model).map(|_| rng.random_range(-0.5..0.5)).collect();
    let router_bias_f32: Vec<f32> = (0..e).map(|_| rng.random_range(-0.1..0.1)).collect();

    // Convert router weights/bias to BF16 for GPU
    let router_weight_bf16: Vec<bf16> = router_weight_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let router_bias_bf16: Vec<bf16> = router_bias_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    // Generate random expert weights and biases
    let w13_len = e * d_model * 2 * d_ff;
    let w2_len = e * d_ff * d_model;

    // Generate W13 in original layout [E, d_model, 2*d_ff] for CPU reference
    let w13_cpu: Vec<bf16> = (0..w13_len).map(|_| bf16::from_f32(rng.random_range(-0.5..0.5))).collect();

    // Transpose to GPU layout [E, 2*d_ff, d_model]
    let mut w13_gpu = vec![bf16::from_f32(0.0); w13_len];
    for expert in 0..e {
        let src_offset = expert * d_model * 2 * d_ff;
        let dst_offset = expert * 2 * d_ff * d_model;
        for dm in 0..d_model {
            for ff in 0..(2 * d_ff) {
                let src_idx = src_offset + dm * 2 * d_ff + ff;
                let dst_idx = dst_offset + ff * d_model + dm;
                w13_gpu[dst_idx] = w13_cpu[src_idx];
            }
        }
    }

    // Generate W2 in original layout [E, d_ff, d_model] for CPU reference
    let w2_cpu: Vec<bf16> = (0..w2_len).map(|_| bf16::from_f32(rng.random_range(-0.5..0.5))).collect();

    // Transpose W2 to GPU layout [E, d_model, d_ff]
    let mut w2_gpu = vec![bf16::from_f32(0.0); w2_len];
    for expert in 0..e {
        let expert_offset = expert * d_ff * d_model;
        for ff in 0..d_ff {
            for dm in 0..d_model {
                // src: [E, d_ff, d_model] -> index: expert_offset + ff * d_model + dm
                // dst: [E, d_model, d_ff] -> index: expert_offset + dm * d_ff + ff
                w2_gpu[expert_offset + dm * d_ff + ff] = w2_cpu[expert_offset + ff * d_model + dm];
            }
        }
    }
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff).map(|_| bf16::from_f32(rng.random_range(-0.1..0.1))).collect();
    let down_biases: Vec<bf16> = (0..e * d_model).map(|_| bf16::from_f32(rng.random_range(-0.1..0.1))).collect();

    // Create Metal buffers
    let x_buf = alloc_allocation_with_data::<B, bf16>(&ctx, &x);
    let router_w_buf = alloc_buffer_with_data::<B, bf16>(&ctx, &router_weight_bf16);
    let router_b_buf = alloc_buffer_with_data::<B, bf16>(&ctx, &router_bias_bf16);
    let w13_buf = alloc_buffer_with_data::<B, bf16>(&ctx, &w13_gpu);
    let w2_buf = alloc_buffer_with_data::<B, bf16>(&ctx, &w2_gpu);
    let up_biases_buf = alloc_buffer_with_data::<B, bf16>(&ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data::<B, bf16>(&ctx, &down_biases);

    // Allocate intermediate buffers (max capacity)
    let max_sumk = t * k;
    let mut topk_ids_buf = alloc_allocation::<B, i32>(&ctx, t * k);
    let mut topk_probs_buf = alloc_allocation::<B, bf16>(&ctx, t * k);
    let mut offsets_buf = alloc_allocation::<B, u32>(&ctx, e + 1);
    let mut sumk_buf = alloc_allocation::<B, u32>(&ctx, 1);
    let num_blocks = ((t + 255) / 256).max(1);
    let num_tiles = ((e + 512 - 1) / 512).max(1);
    let entries = num_blocks * num_tiles * 512usize;
    let mut partials_buf = alloc_allocation::<B, u32>(&ctx, entries);
    let mut block_bases_buf = alloc_allocation::<B, u32>(&ctx, entries);
    let mut block_alloc_buf = alloc_allocation::<B, u32>(&ctx, entries);
    let mut bucketed_ids_buf = alloc_allocation::<B, i32>(&ctx, max_sumk);
    let mut bucketed_probs_buf = alloc_allocation::<B, bf16>(&ctx, max_sumk);
    let mut tok2row_buf = alloc_allocation::<B, i32>(&ctx, t * k);
    let mut x_perm_buf = alloc_allocation::<B, bf16>(&ctx, max_sumk * d_model);
    let mut y_partial_buf = alloc_allocation::<B, bf16>(&ctx, max_sumk * d_model);
    let mut y_out_buf = alloc_allocation::<B, bf16>(&ctx, t * d_model);
    const BLOCK_M_DECODE: usize = 4; // matches two-pass decode kernel configuration
    let h_blocks_decode = (d_ff + BLOCK_M_DECODE - 1) / BLOCK_M_DECODE;
    let max_tiles = max_sumk * h_blocks_decode;
    let mut tile_counts_buf = alloc_allocation::<B, u32>(&ctx, e);
    let mut tile_offsets_buf = alloc_allocation::<B, u32>(&ctx, e + 1);
    let mut tile_map_buf = alloc_allocation::<B, u32>(&ctx, max_tiles * 3);
    let mut total_tiles_buf = alloc_allocation::<B, u32>(&ctx, 8);
    let mut dispatch_args_buf = alloc_allocation::<B, u32>(&ctx, 3);

    // Encode ALL kernels in one command buffer
    eprintln!("[E2E] Encoding entire MoE pipeline in single command buffer...");
    let mut encoder = Encoder::new(ctx).expect("Failed to create encoder");

    // Router + TopK (fused kernel)
    let router_topk = <B::Kernels as Kernels>::MoeRouterTopKKernel::new(&ctx, DataType::BF16).expect("router+topk");
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

    let fused_kernel = <B::Kernels as Kernels>::MoeCountsOffsetsFusedKernel::new(&ctx).expect("fused kernel");
    fused_kernel.encode(
        &topk_ids_buf,
        &mut offsets_buf,
        &mut sumk_buf,
        &mut partials_buf,
        t as u32,
        e as u32,
        k as u32,
        &mut encoder,
    );

    let scatter_bases_kernel =
        <B::Kernels as Kernels>::MoeBlockBasesFromPartialsKernel::new(&ctx).expect("scatter bases kernel");
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

    let scatter_map_kernel = <B::Kernels as Kernels>::MoeScatterBucketsMapKernel::new(&ctx, DataType::BF16)
        .expect("Failed to create <<Metal as Backend>::Kernels as Kernels>::MoeScatterBucketsMapKernel");
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

    let gather = MoeGatherKernels::<B>::new(&ctx).expect("gather");
    gather.encode(
        &mut encoder,
        DataType::BF16,
        MoeGatherArguments {
            x: &x_buf,
            bucketed_ids: &bucketed_ids_buf,
            x_perm: &mut x_perm_buf,
            sumk: &sumk_buf,
            t,
            k,
            d_model,
        },
    );

    // Additional buffers for 2-pass
    let total_rows = t * k;
    let mut hidden_buf = alloc_allocation::<B, f32>(&ctx, total_rows * d_ff);
    let mut row_expert_map_buf = alloc_allocation::<B, u32>(&ctx, total_rows);

    let experts = MoeExpertsTwoPassPrefillBlock::<B>::new(&ctx).expect("experts");
    let num_tiles_k = ((d_ff + 64 - 1) / 64) as u32;
    let args = MoeExpertsTwoPassArguments {
        x_perm: &x_perm_buf,
        expert_offsets: &offsets_buf,
        row_expert_map: &mut row_expert_map_buf,
        hidden: &mut hidden_buf,
        output: &mut y_partial_buf,
        w13_all: &w13_buf,
        w2_all: &w2_buf,
        up_biases: &up_biases_buf,
        down_biases: &down_biases_buf,
        tile_counts: &mut tile_counts_buf,
        tile_offsets: &mut tile_offsets_buf,
        tile_map: &mut tile_map_buf,
        total_tiles: &mut total_tiles_buf,
        dispatch_args: &mut dispatch_args_buf,
        total_rows,
        d_model,
        d_ff,
        e,
        num_tiles_k,
        gating_code,
        gate_clip_min: gate_clip.0,
        gate_clip_max: gate_clip.1,
        up_clip_min: up_clip.0,
        up_clip_max: up_clip.1,
        silu_alpha,
        data_type: DataType::BF16,
    };
    experts.encode(&mut encoder, args);

    let finalize = <B::Kernels as Kernels>::MoeFinalizeKernel::new(&ctx, DataType::BF16).expect("finalize");
    finalize.encode(
        &tok2row_buf,
        &topk_probs_buf,
        &y_partial_buf,
        &mut y_out_buf,
        t as u32,
        d_model as u32,
        k as u32,
        &mut encoder,
    );

    eprintln!("[E2E] All kernels encoded. Committing ONCE and waiting...");
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    eprintln!("[E2E] GPU execution completed");

    // Read GPU output
    let y_out_bf16 = allocation_prefix_to_vec::<B, bf16>(&y_out_buf, t * d_model);
    let y_out_gpu: Vec<f32> = y_out_bf16.iter().map(|&v| f32::from(v)).collect();

    // Validate GPU output is finite
    let nan_count = y_out_gpu.iter().filter(|v| v.is_nan()).count();
    let inf_count = y_out_gpu.iter().filter(|v| v.is_infinite()).count();

    eprintln!("[E2E] GPU output stats: nan={}, inf={}", nan_count, inf_count);
    assert_eq!(nan_count, 0, "GPU output contains {} NaN values", nan_count);
    assert_eq!(inf_count, 0, "GPU output contains {} Inf values", inf_count);

    // Run CPU reference (router logits already computed above)
    // NOTE: Keep weights as BF16 and convert per-access to match GPU precision
    eprintln!("[{}] Running CPU reference implementation...", test_name);

    // Debug: Verify data consistency and tile bookkeeping for large-scale tests
    if d_model >= 512 {
        eprintln!(
            "[E2E] Large-scale debug: x[0]={:.6}, w13[0]={:.6}, w2[0]={:.6}",
            f32::from(x[0]),
            f32::from(w13_cpu[0]),
            f32::from(w2_cpu[0])
        );

        // Probe tile bookkeeping
        let total_tiles_cpu = allocation_prefix_to_vec::<B, u32>(&total_tiles_buf, 2);
        let dispatch_args_cpu = allocation_prefix_to_vec::<B, u32>(&dispatch_args_buf, 3);
        let tile_offsets_cpu = allocation_prefix_to_vec::<B, u32>(&tile_offsets_buf, (e + 1).min(8));
        let sumk_val = allocation_to_vec::<B, u32>(&sumk_buf)[0] as usize;
        eprintln!("[E2E] Tile bookkeeping:");
        eprintln!(
            "[E2E]   total_tiles={}, dispatch_args=({}, {}, {})",
            total_tiles_cpu[0], dispatch_args_cpu[0], dispatch_args_cpu[1], dispatch_args_cpu[2]
        );
        eprintln!("[E2E]   tile_offsets[0..{}]={:?}", (e + 1).min(8), &tile_offsets_cpu);
        eprintln!("[E2E]   sumk={}, num_tiles_k={}", sumk_val, num_tiles_k);

        // For multi-token tests with large d_ff, verify gather output (x_perm)
        if t > 1 && d_ff >= 256 {
            let x_perm_cpu = allocation_prefix_to_vec::<B, bf16>(&x_perm_buf, sumk_val * d_model);
            eprintln!("[E2E] x_perm diagnostics (sumk={}):", sumk_val);
            eprintln!("[E2E]   Row 0 [0:8]: {:?}", &x_perm_cpu[0..8].iter().map(|&v| f32::from(v)).collect::<Vec<_>>());
            if sumk_val > 1 {
                eprintln!(
                    "[E2E]   Row 1 [{}:{}]: {:?}",
                    d_model,
                    d_model + 8,
                    &x_perm_cpu[d_model..d_model + 8].iter().map(|&v| f32::from(v)).collect::<Vec<_>>()
                );
            }

            // Check tile_map for first few tiles
            let tile_map_cpu = allocation_prefix_to_vec::<B, u32>(&tile_map_buf, 12.min(max_tiles * 3));
            eprintln!("[E2E] tile_map (first 4 tiles): {:?}", &tile_map_cpu);

            // CRITICAL: Check tok2row mapping for multi-token tests
            let tok2row_cpu = allocation_prefix_to_vec::<B, i32>(&tok2row_buf, t * k);
            eprintln!("[E2E] tok2row[0..{}]: {:?}", t * k, &tok2row_cpu);
            eprintln!(
                "[E2E]   Expected: token 0→row {}, token 1→row {}",
                0,
                if sumk_val > 1 {
                    1
                } else {
                    -1
                }
            );

            // Check y_partial at specific indices where finalize reads for token 1
            let y_partial_full = allocation_prefix_to_vec::<B, bf16>(&y_partial_buf, sumk_val * d_model);
            eprintln!("[E2E] y_partial spot check:");
            eprintln!("[E2E]   y_partial[48] (row 0, col 48) = {:.6}", f32::from(y_partial_full[48]));
            if sumk_val > 1 {
                // Check positions where mismatches occur (odd indices near tile boundaries)
                eprintln!("[E2E]   Row 1 positions (even=working, odd=corrupted?):");
                for &pos in &[48, 56, 57, 58, 59, 60, 61, 62, 63, 64, 120, 121, 122, 123, 124, 125, 126, 127, 128] {
                    let idx = 1 * d_model + pos;
                    if idx < y_partial_full.len() {
                        eprintln!("[E2E]     y_partial[row1, col {}] = {:.3}", pos, f32::from(y_partial_full[idx]));
                    }
                }
            }
        }
    }

    let y_cpu = moe_cpu_reference(
        &x,
        &router_weight_f32,
        &router_bias_f32,
        &w13_cpu,
        &w2_cpu,
        &up_biases,
        &down_biases,
        t,
        e,
        k,
        d_model,
        d_ff,
        gating_code,
        silu_alpha,
        gate_clip,
        up_clip,
    );

    // Compare GPU vs CPU
    eprintln!("[E2E] Comparing GPU vs CPU outputs...");
    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut total_abs_error = 0.0f32;
    let mut mismatches = 0;

    for i in 0..(t * d_model) {
        let gpu_val = y_out_gpu[i];
        let cpu_val = y_cpu[i];
        let abs_diff = (gpu_val - cpu_val).abs();
        let rel_diff = if cpu_val.abs() > 1e-6 {
            abs_diff / cpu_val.abs()
        } else {
            abs_diff
        };

        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
        total_abs_error += abs_diff;

        let threshold_rel = 0.1;
        let threshold_abs = 1e-3;

        if rel_diff > threshold_rel && abs_diff > threshold_abs {
            let print_limit = if test_name.contains("BoundarySweep_D1024_FF256_T2") {
                usize::MAX // Print ALL for debugging
            } else {
                10
            };
            if mismatches < print_limit {
                eprintln!(
                    "[{}]   Mismatch idx {}: GPU={:.6}, CPU={:.6}, abs={:.6}, rel={:.6}",
                    test_name, i, gpu_val, cpu_val, abs_diff, rel_diff
                );
            }
            mismatches += 1;
        }
    }

    let mean_abs_error = total_abs_error / (t * d_model) as f32;

    eprintln!(
        "[E2E] Error metrics: max_abs={:.6}, max_rel={:.6}, mean_abs={:.6}, mismatches={}",
        max_abs_diff, max_rel_diff, mean_abs_error, mismatches
    );

    // Debug: print some sample values from intermediate buffers
    eprintln!("[E2E] === DEBUG: Intermediate values ===");
    let topk_ids_gpu = allocation_prefix_to_vec::<B, i32>(&topk_ids_buf, t * k);
    let topk_probs_gpu = allocation_prefix_to_vec::<B, bf16>(&topk_probs_buf, t * k);
    eprintln!("[E2E] TopK IDs: {:?}", topk_ids_gpu);
    eprintln!("[E2E] TopK Probs: {:?}", topk_probs_gpu.iter().map(|&v| f32::from(v)).collect::<Vec<_>>());

    let y_partial_gpu = allocation_prefix_to_vec::<B, bf16>(&y_partial_buf, max_sumk * d_model);
    let sumk_actual = allocation_to_vec::<B, u32>(&sumk_buf)[0] as usize;
    eprintln!("[E2E] sumk={}", sumk_actual);
    let sample_size = 16.min(sumk_actual * d_model);
    eprintln!(
        "[E2E] y_partial sample (first {}): {:?}",
        sample_size,
        &y_partial_gpu[..sample_size].iter().map(|&v| f32::from(v)).collect::<Vec<_>>()
    );

    // For multi-row debugging: print both rows
    if sumk_actual > 1 && d_ff >= 256 {
        eprintln!(
            "[E2E] y_partial row 1 [{}-{}]: {:?}",
            d_model,
            d_model + 16,
            &y_partial_gpu[d_model..d_model + 16].iter().map(|&v| f32::from(v)).collect::<Vec<_>>()
        );
    }
    let out_sample_size = 16.min(t * d_model);
    eprintln!("[E2E] y_out sample (token 0): {:?}", &y_out_gpu[..out_sample_size]);
    if t > 1 && d_model >= 512 {
        eprintln!(
            "[E2E] y_out sample (token 1, first 32): {:?}",
            &y_out_gpu[d_model..d_model + 32].iter().map(|&v| f32::from(v)).collect::<Vec<_>>()
        );
    }
    eprintln!("[E2E] CPU ref sample (token 0): {:?}", &y_cpu[..out_sample_size]);
    if t > 1 && d_model >= 512 {
        eprintln!("[E2E] CPU ref sample (token 1): {:?}", &y_cpu[d_model..d_model + 16]);

        // Compare token 1 outputs element-wise (more positions to find pattern)
        let gpu_t1 = &y_out_gpu[d_model..];
        let cpu_t1 = &y_cpu[d_model..];
        eprintln!("[E2E] Token 1 detailed comparison:");
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 48, 64, 121, 128].iter() {
            if *i < d_model {
                let diff = (f32::from(gpu_t1[*i]) - cpu_t1[*i]).abs();
                eprintln!("[E2E]   [{}]: GPU={:.6}, CPU={:.6}, diff={:.6}", i, f32::from(gpu_t1[*i]), cpu_t1[*i], diff);
            }
        }
    }

    let k0_iters = (d_model + 31) / 32;
    let ff0_iters = (d_ff + 31) / 32;
    let product = k0_iters * ff0_iters;
    let sqrt_product = (product as f32).sqrt();

    let multi_expert_drift = 0.03 * sqrt_product * (k as f32 / 2.0);
    let mean_abs_threshold = 0.1f32.max(multi_expert_drift);

    if mean_abs_error >= mean_abs_threshold {
        panic!("[{}] Mean absolute error {:.6} exceeds threshold {:.6}", test_name, mean_abs_error, mean_abs_threshold);
    }

    // Warn on high max_rel but don't fail (can be inflated by near-zero values)
    if max_rel_diff > 1.0 && mismatches > (t * d_model / 4) {
        eprintln!(
            "[{}] WARNING: High max_rel={:.2} with {} mismatches, but mean_abs={:.6} is acceptable",
            test_name, max_rel_diff, mismatches, mean_abs_error
        );
    }

    eprintln!("[{}] ✓ PASSED (GPU matches CPU reference)", test_name);
}

// Test 1: Minimal (K=1, small dims, no clipping, alpha=1.0)
#[test]
fn test_moe_minimal() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            1,                                  // t
            2,                                  // e
            1,                                  // k
            4,                                  // d_model
            4,                                  // d_ff
            2,                                  // gating_code (SwiGLU)
            1.0,                                // silu_alpha
            (f32::NEG_INFINITY, f32::INFINITY), // gate_clip
            (f32::NEG_INFINITY, f32::INFINITY), // up_clip
            0xE2E_0001,
            "Minimal_K1_SwiGLU",
        );
    })
}

// Test 2: Multi-expert (K=2, tests finalize weighted sum)
#[test]
fn test_moe_multi_expert() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            2,                        // t
            4,                        // e
            2,                        // k
            8,                        // d_model
            8,                        // d_ff
            2,                        // gating_code (SwiGLU)
            1.702,                    // silu_alpha (GPT-OSS value)
            (f32::NEG_INFINITY, 7.0), // gate_clip (GPT-OSS config)
            (-6.0, 8.0),              // up_clip (GPT-OSS config)
            0xE2E_0002,
            "Multi_K2_SwiGLU",
        );
    })
}

// Test 3: Large d_model only (K=1 to isolate from finalize complexity)
#[test]
fn test_moe_large_d_model_only() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            1,  // t
            2,  // e
            1,  // k
            64, // d_model (tests k0 chunking)
            8,  // d_ff (single ff0 chunk)
            2,  // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0003,
            "LargeDModel64",
        );
    })
}

// Test 4a: d_ff=32 (exactly 1 full chunk - baseline for multi-chunk)
#[test]
fn test_1_full_chunk() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            1,  // t
            2,  // e
            1,  // k
            8,  // d_model
            32, // d_ff (exactly 1 full chunk)
            2,  // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0004,
            "LargeFF32_SingleChunk",
        );
    })
}

// Test 4b: d_ff=48 (1.5 chunks - tests partial second chunk)
#[test]
fn test_1_5_chunks() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            1,  // t
            2,  // e
            1,  // k
            8,  // d_model
            48, // d_ff (1 full + 1 partial chunk)
            2,  // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0005,
            "LargeFF48_Partial",
        );
    })
}

// Test 4c: d_ff=64 (exactly 2 full chunks)
#[test]
fn test_2_full_chunks() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            1,  // t
            2,  // e
            1,  // k
            8,  // d_model
            64, // d_ff (exactly 2 full chunks)
            2,  // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0006,
            "LargeFF64_TwoChunks",
        );
    })
}

// Test 5: GELU activation (gating_code=0)
#[test]
fn test_gelu_activation() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            1,   // t
            2,   // e
            1,   // k
            8,   // d_model
            16,  // d_ff
            0,   // gating_code (GELU)
            1.0, // silu_alpha (unused for GELU)
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0007,
            "GELU_Activation",
        );
    })
}

// Test 6: SiLU activation (gating_code=1)
#[test]
fn test_silu_activation() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            1,  // t
            2,  // e
            1,  // k
            8,  // d_model
            16, // d_ff
            1,  // gating_code (SiLU)
            1.702,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0008,
            "SiLU_Activation",
        );
    })
}

// Test 7: GEGLU activation (gating_code=3)
#[test]
fn test_geglu_activation() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            1,   // t
            2,   // e
            1,   // k
            8,   // d_model
            16,  // d_ff
            3,   // gating_code (GEGLU)
            1.0, // silu_alpha (unused for GEGLU)
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0009,
            "GEGLU_Activation",
        );
    })
}

// Test 8: Bucket stress - multiple tokens, larger E, stress scatter/gather
#[test]
fn test_bucket_stress() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            8,  // t (8 tokens)
            8,  // e (8 experts)
            2,  // k (each token picks 2 experts)
            16, // d_model
            32, // d_ff
            2,  // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0010,
            "BucketStress_T8_E8_K2",
        );
    })
}

// Test 9a: K=4 small-scale to verify multi-expert accumulation
#[test]
fn test_small_scale() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            2,  // t
            4,  // e
            4,  // k (all experts)
            8,  // d_model
            16, // d_ff
            2,  // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0011,
            "Multi_K4_Small",
        );
    })
}

// Test 9b: d_model=128 (4 k0 chunks, 2 n-tiles)
#[test]
fn test_model_128() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            2,   // t
            4,   // e
            1,   // k
            128, // d_model (4 k0 chunks, 2 n-tiles)
            32,  // d_ff
            2,   // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0012,
            "MultiNTile_D128",
        );
    })
}

// Test 9c: d_model=192 (6 k0 chunks, 3 n-tiles - test threshold)
#[test]
fn test_model_192() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            2,   // t
            4,   // e
            1,   // k
            192, // d_model (6 k0 chunks, 3 n-tiles)
            32,  // d_ff
            2,   // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0013,
            "MultiNTile_D192_3tiles",
        );
    })
}

// Test 9d: d_model=256 (8 k0 chunks, 4 n-tiles)
#[test]
fn test_model_256() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            2,   // t
            4,   // e
            1,   // k
            256, // d_model (8 k0 chunks, 4 n-tiles)
            32,  // d_ff
            2,   // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0014,
            "MultiNTile_D256_4tiles",
        );
    })
}

// Test 10a: Isolate large d_ff with small d_model (FF accumulation test)
#[test]
fn test_ff_accumulation() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            2,     // t
            4,     // e
            1,     // k
            64,    // d_model (small, 1 n-tile, 2 k0 chunks)
            14336, // d_ff (448 ff0 chunks!) - production scale
            2,     // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0016,
            "IsolateFF_D64_FF14336_K1",
        );
    })
}

// Test 10b: d_model=1024 (16 n-tiles)
#[test]
fn test_model_1024() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            2,    // t
            4,    // e
            1,    // k
            1024, // d_model (32 k0 chunks, 16 n-tiles)
            64,   // d_ff (small)
            2,    // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0017,
            "IsolateD_D1024_FF64_K1",
        );
    })
}

// Test 10c: Verify layout with d_model=96, d_ff=96 (3 chunks each)
// This tests non-power-of-2 to catch stride bugs
#[test]
fn test_stride() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            1,  // t (single token)
            2,  // e
            1,  // k
            96, // d_model (3 k0 chunks - odd number)
            96, // d_ff (3 ff0 chunks - odd number)
            2,  // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0018,
            "VerifyLayout_D96_FF96",
        );
    })
}

// Test 10d: d_model=1536
#[test]
fn test_model_1536() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            2,    // t
            4,    // e
            1,    // k
            1536, // d_model (48 k0 chunks, 24 n-tiles)
            64,   // d_ff (small)
            2,    // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0019,
            "IsolateD_D1536_FF64_K1",
        );
    })
}

// Test 10e: 2 n-tiles
#[test]
fn test_2n_tiles() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            1,   // t
            2,   // e
            1,   // k
            128, // d_model (2 n-tiles)
            256, // d_ff
            2,   // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0020,
            "NTileTest_D128_2tiles",
        );
    })
}

// Test 10g: 8 n-tiles - find threshold
#[test]
fn test_8n_tiles() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            1,   // t
            2,   // e
            1,   // k
            512, // d_model (8 n-tiles)
            256, // d_ff
            2,   // gating_code (SwiGLU)
            1.0,
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
            0xE2E_0021,
            "NTileTest_D512_8tiles",
        );
    })
}

// Test 11b: Production scale with clipping but K=1
#[test]
#[ignore]
fn test_prod_scale_clipping() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            2,                         // t
            8,                         // e
            1,                         // k
            4096,                      // d_model
            14336,                     // d_ff
            2,                         // gating_code (SwiGLU)
            1.702,                     // silu_alpha (GPT-OSS value)
            (f32::NEG_INFINITY, 20.0), // gate_clip_max
            (-19.0, 21.0),             // up_clip
            0xE2E_0029,
            "ProductionScale_D4096_H14336_E8_K1",
        );
    })
}

// Test 11c: PRODUCTION SCALE T=1 decode (triggers GEMV v2)
#[test]
#[ignore]
fn test_prod_scale_decode() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            1,                         // t (single token decode)
            16,                        // e (16 experts)
            2,                         // k (2 experts per token)
            4096,                      // d_model
            14336,                     // d_ff
            2,                         // gating_code (SwiGLU)
            1.702,                     // silu_alpha (GPT-OSS value)
            (f32::NEG_INFINITY, 20.0), // gate_clip_max (GPT-OSS config)
            (-19.0, 21.0),             // up_clip (GPT-OSS config)
            0xE2E_0030,
            "ProductionScale_T1_D4096_H14336_E16_K2",
        );
    })
}

// Test 11d: FULL PRODUCTION SCALE with K=2, T=4 (uses tiled MMA)
#[test]
#[ignore]
fn test_prod_scale_tile_mma() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        run_moe_parity_test::<B>(
            &ctx,
            4,                         // t (4 tokens)
            16,                        // e (16 experts)
            2,                         // k (2 experts per token)
            4096,                      // d_model (128 k0 chunks, 64 n-tiles!)
            14336,                     // d_ff (448 ff0 chunks!)
            2,                         // gating_code (SwiGLU)
            1.702,                     // silu_alpha (GPT-OSS value)
            (f32::NEG_INFINITY, 20.0), // gate_clip_max (GPT-OSS config)
            (-19.0, 21.0),             // up_clip (GPT-OSS config)
            0xE2E_0031,
            "ProductionScale_D4096_H14336_E16_K2",
        );
    })
}
