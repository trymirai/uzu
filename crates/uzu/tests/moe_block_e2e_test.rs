#![cfg(feature = "moe_dev_tests")]

use half::bf16;
use metal::MTLResourceOptions;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    MTLContext,
    kernel::{
        KernelDataType, MoeBlockBasesArguments, MoeBucketCountsArguments,
        MoeBucketCountsKernel, MoeExpertsArguments, MoeExpertsKernel,
        MoeFinalizeArguments, MoeFinalizeKernel, MoeOffsetsScanArguments,
        MoeOffsetsScanKernel, MoeScatterKernels, MoeScatterWithMapArguments,
        MoeTopKArguments, MoeTopKKernel,
        moe::{MoeGatherArguments, MoeGatherKernel},
    },
};

fn create_ctx() -> MTLContext {
    let device = metal::Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    MTLContext::new(device, queue).expect("ctx")
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn moe_cpu_reference(
    x: &[f32],
    router_logits: &[f32],
    w13: &[f32],
    w2: &[f32],
    up_biases: &[f32],
    down_biases: &[f32],
    t: usize,
    e: usize,
    k: usize,
    d_model: usize,
    d_ff: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; t * d_model];

    for token_idx in 0..t {
        let logits_start = token_idx * e;
        let token_logits = &router_logits[logits_start..logits_start + e];

        // TopK selection
        let mut indices_and_logits: Vec<(usize, f32)> =
            token_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indices_and_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_k_indices: Vec<usize> =
            indices_and_logits[..k].iter().map(|&(i, _)| i).collect();
        let top_k_logits: Vec<f32> =
            indices_and_logits[..k].iter().map(|&(_, v)| v).collect();

        // Softmax normalization (matching Python: jax.nn.softmax)
        let max_logit =
            top_k_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 =
            top_k_logits.iter().map(|&v| (v - max_logit).exp()).sum();
        let weights: Vec<f32> = top_k_logits
            .iter()
            .map(|&v| (v - max_logit).exp() / exp_sum)
            .collect();

        // Process each selected expert (matching Python: vmap(run_expert))
        for (expert_idx, &weight) in top_k_indices.iter().zip(weights.iter()) {
            let x_start = token_idx * d_model;
            let token_input = &x[x_start..x_start + d_model];

            // Expert MLP weights layout: W13[E, d_model, 2*d_ff], W2[E, d_ff, d_model]
            let w13_offset = expert_idx * d_model * 2 * d_ff;
            let w2_offset = expert_idx * d_ff * d_model;
            let bias_up_offset = expert_idx * 2 * d_ff;
            let bias_down_offset = expert_idx * d_model;

            // Up projection: W13[expert, input, output] where output=[up, gate]
            let mut up_out = vec![0.0f32; d_ff];
            let mut gate_out = vec![0.0f32; d_ff];

            for ff_idx in 0..d_ff {
                let mut up_sum = up_biases[bias_up_offset + ff_idx];
                let mut gate_sum = up_biases[bias_up_offset + d_ff + ff_idx];

                for input_idx in 0..d_model {
                    let w_base = w13_offset + input_idx * 2 * d_ff;
                    up_sum += token_input[input_idx] * w13[w_base + ff_idx];
                    gate_sum +=
                        token_input[input_idx] * w13[w_base + d_ff + ff_idx];
                }

                up_out[ff_idx] = up_sum;
                gate_out[ff_idx] = gate_sum;
            }

            // SwiGLU: silu(gate) * up (matching Python line 224: down_projection(up_proj * gate))
            let mut hidden = vec![0.0f32; d_ff];
            for i in 0..d_ff {
                hidden[i] = silu(gate_out[i]) * up_out[i];
            }

            // Down projection: W2[expert, ff, output]
            for out_idx in 0..d_model {
                let mut sum = down_biases[bias_down_offset + out_idx];
                for ff_idx in 0..d_ff {
                    sum += hidden[ff_idx]
                        * w2[w2_offset + ff_idx * d_model + out_idx];
                }
                output[x_start + out_idx] += weight * sum;
            }
        }
    }

    output
}

#[test]
fn test_moe_block_end_to_end() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xE2E_0001);

    let t = 1usize;
    let e = 2usize;
    let k = 1usize;
    let d_model = 4usize;
    let d_ff = 4usize;

    eprintln!(
        "[E2E] Setup: T={}, E={}, K={}, d_model={}, d_ff={}",
        t, e, k, d_model, d_ff
    );

    // Generate random input and router logits
    let x: Vec<bf16> = (0..t * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
        .collect();
    let router_logits: Vec<f32> =
        (0..t * e).map(|_| rng.random_range(-2.0..2.0)).collect();

    // Generate random expert weights
    let w13_len = e * d_model * 2 * d_ff;
    let w2_len = e * d_ff * d_model;
    let w13: Vec<bf16> = (0..w13_len)
        .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();
    let w2: Vec<bf16> = (0..w2_len)
        .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();

    // Create Metal buffers
    let x_buf = ctx.device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let logits_buf = ctx.device.new_buffer_with_data(
        router_logits.as_ptr() as *const _,
        (router_logits.len() * std::mem::size_of::<f32>()) as u64,
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
    let up_biases_buf = ctx.device.new_buffer(
        (e * 2 * d_ff * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let down_biases_buf = ctx.device.new_buffer(
        (e * d_model * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::write_bytes(
            up_biases_buf.contents(),
            0,
            e * 2 * d_ff * std::mem::size_of::<bf16>(),
        );
        std::ptr::write_bytes(
            down_biases_buf.contents(),
            0,
            e * d_model * std::mem::size_of::<bf16>(),
        );
    }

    // Allocate intermediate buffers (max capacity)
    let max_sumk = t * k;
    let topk_ids_buf = ctx.device.new_buffer(
        (t * k * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let topk_probs_buf = ctx.device.new_buffer(
        (t * k * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let counts_buf = ctx.device.new_buffer(
        (e * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let offsets_buf = ctx.device.new_buffer(
        ((e + 1) * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sumk_buf = ctx.device.new_buffer(
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let num_blocks = ((t + 255) / 256).max(1);
    let num_tiles = ((e + 512 - 1) / 512).max(1);
    let entries = num_blocks * num_tiles * 512usize;
    let partials_buf = ctx.device.new_buffer(
        (entries * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let block_bases_buf = ctx.device.new_buffer(
        (entries * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let block_alloc_buf = ctx.device.new_buffer(
        (entries * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let bucketed_ids_buf = ctx.device.new_buffer(
        (max_sumk * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let bucketed_probs_buf = ctx.device.new_buffer(
        (max_sumk * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let tok2row_buf = ctx.device.new_buffer(
        (t * k * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let x_perm_buf = ctx.device.new_buffer(
        (max_sumk * d_model * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_partial_buf = ctx.device.new_buffer(
        (max_sumk * d_model * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_out_buf = ctx.device.new_buffer(
        (t * d_model * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    const BM: usize = 16;
    let max_tiles = e * ((max_sumk + BM - 1) / BM);
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

    // Encode ALL kernels in one command buffer
    eprintln!("[E2E] Encoding entire MoE pipeline in single command buffer...");
    let cb = ctx.command_queue.new_command_buffer();

    let topk = MoeTopKKernel::new(&ctx).expect("topk");
    topk.encode(
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
    .expect("topk encode");

    let bucket = MoeBucketCountsKernel::new(&ctx).expect("bucket");
    bucket
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
        .expect("bucket encode");

    let scan = MoeOffsetsScanKernel::new(&ctx).expect("scan");
    scan.encode(
        &cb,
        MoeOffsetsScanArguments {
            counts_buffer: &counts_buf,
            offsets_buffer: &offsets_buf,
            sumk_buffer: &sumk_buf,
            e,
        },
    )
    .expect("scan encode");

    let scatter = MoeScatterKernels::new(&ctx).expect("scatter");
    scatter
        .encode_block_bases(
            &cb,
            MoeBlockBasesArguments {
                partials_buffer: &partials_buf,
                block_bases_buffer: &block_bases_buf,
                block_alloc_buffer: &block_alloc_buf,
                e,
                num_blocks,
                num_tiles,
            },
        )
        .expect("block bases");

    scatter
        .encode_scatter_with_map(
            &cb,
            MoeScatterWithMapArguments {
                base: uzu::backends::metal::kernel::MoeScatterArguments {
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

    let gather = MoeGatherKernel::new(&ctx).expect("gather");
    gather
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

    let experts = MoeExpertsKernel::new(&ctx).expect("experts");
    const BN: usize = 64;
    let num_tiles_n = (d_model + BN - 1) / BN;
    experts
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
                gating_code: 2,
                gate_clip_min: f32::NEG_INFINITY,
                gate_clip_max: f32::INFINITY,
                up_clip_min: f32::NEG_INFINITY,
                up_clip_max: f32::INFINITY,
                silu_alpha: 1.0,
                data_type: KernelDataType::BFloat16,
            },
        )
        .expect("experts encode");

    let finalize = MoeFinalizeKernel::new(&ctx).expect("finalize");
    finalize
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

    eprintln!("[E2E] All kernels encoded. Committing ONCE and waiting...");
    cb.commit();
    cb.wait_until_completed();
    eprintln!("[E2E] GPU execution completed");

    // Read GPU output
    let y_out_bf16 = unsafe {
        std::slice::from_raw_parts(
            y_out_buf.contents() as *const bf16,
            t * d_model,
        )
    };
    let y_out_gpu: Vec<f32> =
        y_out_bf16.iter().map(|&v| f32::from(v)).collect();

    // Validate GPU output is finite
    let nan_count = y_out_gpu.iter().filter(|v| v.is_nan()).count();
    let inf_count = y_out_gpu.iter().filter(|v| v.is_infinite()).count();

    eprintln!("[E2E] GPU output stats: nan={}, inf={}", nan_count, inf_count);
    assert_eq!(nan_count, 0, "GPU output contains {} NaN values", nan_count);
    assert_eq!(inf_count, 0, "GPU output contains {} Inf values", inf_count);

    // Debug: Check what CPU TopK produces
    eprintln!("[E2E] === CPU TopK Debug ===");
    for token_idx in 0..t.min(2) {
        let logits_start = token_idx * e;
        let token_logits = &router_logits[logits_start..logits_start + e];
        let mut indices_and_logits: Vec<(usize, f32)> =
            token_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indices_and_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_k_logits: Vec<f32> =
            indices_and_logits[..k].iter().map(|&(_, v)| v).collect();
        let max_logit =
            top_k_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 =
            top_k_logits.iter().map(|&v| (v - max_logit).exp()).sum();
        let weights: Vec<f32> = top_k_logits
            .iter()
            .map(|&v| (v - max_logit).exp() / exp_sum)
            .collect();
        eprintln!(
            "[E2E]   Token {}: logits={:?}, weights={:?}",
            token_idx, &token_logits, weights
        );
    }

    // Run CPU reference
    eprintln!("[E2E] Running CPU reference implementation...");
    let x_f32: Vec<f32> = x.iter().map(|&v| f32::from(v)).collect();
    let w13_f32: Vec<f32> = w13.iter().map(|&v| f32::from(v)).collect();
    let w2_f32: Vec<f32> = w2.iter().map(|&v| f32::from(v)).collect();
    let up_biases_f32 = vec![0.0f32; e * 2 * d_ff];
    let down_biases_f32 = vec![0.0f32; e * d_model];

    let y_cpu = moe_cpu_reference(
        &x_f32,
        &router_logits,
        &w13_f32,
        &w2_f32,
        &up_biases_f32,
        &down_biases_f32,
        t,
        e,
        k,
        d_model,
        d_ff,
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

        if rel_diff > 0.01 && abs_diff > 1e-3 {
            if mismatches < 5 {
                eprintln!(
                    "[E2E]   Mismatch at idx {}: GPU={:.6}, CPU={:.6}, abs_diff={:.6}, rel_diff={:.6}",
                    i, gpu_val, cpu_val, abs_diff, rel_diff
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
    let topk_ids_gpu = unsafe {
        std::slice::from_raw_parts(topk_ids_buf.contents() as *const i32, t * k)
    };
    let topk_probs_gpu = unsafe {
        std::slice::from_raw_parts(
            topk_probs_buf.contents() as *const bf16,
            t * k,
        )
    };
    eprintln!("[E2E] TopK IDs: {:?}", topk_ids_gpu);
    eprintln!(
        "[E2E] TopK Probs: {:?}",
        topk_probs_gpu.iter().map(|&v| f32::from(v)).collect::<Vec<_>>()
    );

    let y_partial_gpu = unsafe {
        std::slice::from_raw_parts(
            y_partial_buf.contents() as *const bf16,
            max_sumk * d_model,
        )
    };
    let sumk_actual = unsafe { *(sumk_buf.contents() as *const u32) } as usize;
    eprintln!("[E2E] sumk={}", sumk_actual);
    let sample_size = 16.min(sumk_actual * d_model);
    eprintln!(
        "[E2E] y_partial sample (first {}): {:?}",
        sample_size,
        &y_partial_gpu[..sample_size]
            .iter()
            .map(|&v| f32::from(v))
            .collect::<Vec<_>>()
    );
    let out_sample_size = 16.min(t * d_model);
    eprintln!("[E2E] y_out sample: {:?}", &y_out_gpu[..out_sample_size]);
    eprintln!("[E2E] CPU ref sample: {:?}", &y_cpu[..out_sample_size]);

    // Assert reasonable accuracy (allowing for bf16 precision loss)
    assert!(
        max_rel_diff < 0.01,
        "Max relative error {:.6} exceeds threshold 0.01",
        max_rel_diff
    );
    assert!(
        mean_abs_error < 0.001,
        "Mean absolute error {:.6} exceeds threshold 0.001",
        mean_abs_error
    );

    eprintln!(
        "[E2E] ✓ MoE block end-to-end test PASSED (GPU matches CPU reference)"
    );
}
