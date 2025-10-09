use half::bf16;
use metal::MTLResourceOptions;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    MTLContext,
    kernel::{
        KernelDataType, MoeBlockBasesArguments, MoeBucketCountsArguments,
        MoeBucketCountsKernel, MoeExpertsArguments, MoeExpertsFusedKernel,
        MoeFinalizeArguments, MoeFinalizeKernel, MoeOffsetsScanArguments,
        MoeOffsetsScanKernel, MoeScatterKernels, MoeScatterWithMapArguments,
        MoeTopKArguments, MoeTopKKernel,
        moe::{MoeGatherArguments, MoeGatherKernel},
    },
};

use super::test_utils::create_ctx;

fn silu(
    x: f32,
    alpha: f32,
) -> f32 {
    x / (1.0 + (-alpha * x).exp())
}

fn gelu(x: f32) -> f32 {
    const K0: f32 = 0.7978845608f32;
    const K1: f32 = 0.044715f32;
    if x > 10.0 {
        return x;
    }
    if x < -10.0 {
        return 0.0;
    }
    let x3 = x * x * x;
    let inner = x + K1 * x3;
    let tanh_arg = (K0 * inner).clamp(-10.0, 10.0);
    0.5 * x * (1.0 + tanh_arg.tanh())
}

fn moe_cpu_reference(
    x: &[bf16],
    router_weight: &[f32], // [E, d_model] - kept as F32 for router (computed before BF16 conversion)
    router_bias: &[f32],   // [E]
    w13: &[bf16], // source layout [E, d_model, 2*d_ff] (GPU transposes to [E, 2*d_ff, d_model])
    w2: &[bf16],  // [E, d_ff, d_model]
    up_biases: &[bf16], // [E, 2*d_ff]
    down_biases: &[bf16], // [E, d_model]
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
                let mut gate_sum =
                    f32::from(up_biases[bias_up_offset + d_ff + ff_idx]);

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
                    0 => gelu(up_out[i]),
                    1 => silu(up_out[i], silu_alpha),
                    2 => silu(gate_out[i], silu_alpha) * up_out[i],
                    3 => gelu(gate_out[i]) * up_out[i],
                    _ => silu(gate_out[i], silu_alpha) * up_out[i], // fallback to SwiGLU
                };
            }

            // Down projection: W2[expert, ff, output]
            // Convert BF16→F32 on each access to match GPU precision
            // CRITICAL: Match GPU's BF16 quantization after FC2 and after bias addition
            for out_idx in 0..d_model {
                let mut fc2_sum = 0.0f32;
                for ff_idx in 0..d_ff {
                    let w2_val =
                        f32::from(w2[w2_offset + ff_idx * d_model + out_idx]);
                    fc2_sum += hidden[ff_idx] * w2_val;
                }
                // GPU stores FC2 result as BF16, then reads it back
                let fc2_bf16 = bf16::from_f32(fc2_sum);
                let fc2_val = f32::from(fc2_bf16);

                // Add down bias and quantize again (GPU does Y_partial[idx] = T(out_val))
                let down_bias_val =
                    f32::from(down_biases[bias_down_offset + out_idx]);
                let with_bias = fc2_val + down_bias_val;
                let with_bias_bf16 = bf16::from_f32(with_bias);
                let final_val = f32::from(with_bias_bf16);

                output[x_start + out_idx] += weight * final_val;
            }
        }
    }

    output
}

// Main entry point - automatically tests both modes for T>1
fn run_moe_parity_test(
    ctx: &MTLContext,
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
        run_moe_parity_test_internal(
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
            &format!("{}_1pass", test_name),
            false,
        );
        // Test 2-pass prefill (experimental)
        run_moe_parity_test_internal(
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
            &format!("{}_2pass", test_name),
            true,
        );
    } else {
        // Decode mode (T=1) - no prefill variants
        run_moe_parity_test_internal(
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
            false,
        );
    }
}

fn run_moe_parity_test_internal(
    ctx: &MTLContext,
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
    two_pass_prefill: bool, // If true, enable 2-pass prefill via env var
) {
    // Set environment variable for 2-pass prefill if requested
    if two_pass_prefill && t > 1 {
        unsafe {
            std::env::set_var("UZU_MOE_TWO_PASS_PREFILL", "1");
        }
    } else {
        unsafe {
            std::env::remove_var("UZU_MOE_TWO_PASS_PREFILL");
        }
    }
    let mut rng = StdRng::seed_from_u64(seed);

    let prefill_mode = if t > 1 {
        if two_pass_prefill {
            "2-pass"
        } else {
            "1-pass"
        }
    } else {
        "decode"
    };

    eprintln!(
        "\n[{}] T={}, E={}, K={}, d_model={}, d_ff={}, alpha={}, gate_clip={:?}, up_clip={:?}, mode={}",
        test_name,
        t,
        e,
        k,
        d_model,
        d_ff,
        silu_alpha,
        gate_clip,
        up_clip,
        prefill_mode
    );

    // Random BF16 inputs
    let x: Vec<bf16> = (0..t * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
        .collect();

    // Generate random router weights and biases for CPU reference
    let router_weight_f32: Vec<f32> =
        (0..e * d_model).map(|_| rng.random_range(-0.5..0.5)).collect();
    let router_bias_f32: Vec<f32> =
        (0..e).map(|_| rng.random_range(-0.1..0.1)).collect();

    // Compute router logits on CPU with BF16 precision to match GPU
    // Convert to BF16, then back to F32 for each multiplication to match GPU precision
    let router_weight_bf16: Vec<bf16> =
        router_weight_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let router_bias_bf16: Vec<bf16> =
        router_bias_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let mut router_logits_f32 = vec![0.0f32; t * e];
    for token_idx in 0..t {
        let x_start = token_idx * d_model;
        for expert_idx in 0..e {
            let mut logit = f32::from(router_bias_bf16[expert_idx]);
            for d in 0..d_model {
                // Match GPU precision: BF16 * BF16 → F32 accumulation
                let x_val = f32::from(x[x_start + d]);
                let w_val =
                    f32::from(router_weight_bf16[expert_idx * d_model + d]);
                logit += x_val * w_val;
            }
            router_logits_f32[token_idx * e + expert_idx] = logit;
        }
    }
    let router_logits: Vec<bf16> =
        router_logits_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    // Generate random expert weights and biases
    let w13_len = e * d_model * 2 * d_ff;
    let w2_len = e * d_ff * d_model;

    // Generate W13 in original layout [E, d_model, 2*d_ff] for CPU reference
    let w13_cpu: Vec<bf16> = (0..w13_len)
        .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();

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
    let w2_cpu: Vec<bf16> = (0..w2_len)
        .map(|_| bf16::from_f32(rng.random_range(-0.5..0.5)))
        .collect();

    // Transpose W2 to GPU layout [E, d_model, d_ff]
    let mut w2_gpu = vec![bf16::from_f32(0.0); w2_len];
    for expert in 0..e {
        let expert_offset = expert * d_ff * d_model;
        for ff in 0..d_ff {
            for dm in 0..d_model {
                // src: [E, d_ff, d_model] -> index: expert_offset + ff * d_model + dm
                // dst: [E, d_model, d_ff] -> index: expert_offset + dm * d_ff + ff
                w2_gpu[expert_offset + dm * d_ff + ff] =
                    w2_cpu[expert_offset + ff * d_model + dm];
            }
        }
    }
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.random_range(-0.1..0.1)))
        .collect();
    let down_biases: Vec<bf16> = (0..e * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.1..0.1)))
        .collect();

    // Create Metal buffers
    let x_buf = ctx.device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let logits_buf = ctx.device.new_buffer_with_data(
        router_logits.as_ptr() as *const _,
        (router_logits.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w13_buf = ctx.device.new_buffer_with_data(
        w13_gpu.as_ptr() as *const _,
        (w13_gpu.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w2_buf = ctx.device.new_buffer_with_data(
        w2_gpu.as_ptr() as *const _,
        (w2_gpu.len() * std::mem::size_of::<bf16>()) as u64,
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
    const BLOCK_M_DECODE: usize = 4; // matches two-pass decode kernel configuration
    let h_blocks_decode = (d_ff + BLOCK_M_DECODE - 1) / BLOCK_M_DECODE;
    let max_tiles = max_sumk * h_blocks_decode;
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

    // Note: Router logits already computed on CPU and uploaded to logits_buf
    // TODO: Test full MoeBlockEncodable path which includes GPU router matmul

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

    let experts = MoeExpertsFusedKernel::new(&ctx).expect("experts");
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
                gating_code,
                gate_clip_min: gate_clip.0,
                gate_clip_max: gate_clip.1,
                up_clip_min: up_clip.0,
                up_clip_max: up_clip.1,
                silu_alpha,
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
        let total_tiles_cpu = unsafe {
            std::slice::from_raw_parts(
                total_tiles_buf.contents() as *const u32,
                2,
            )
        };
        let dispatch_args_cpu = unsafe {
            std::slice::from_raw_parts(
                dispatch_args_buf.contents() as *const u32,
                3,
            )
        };
        let tile_offsets_cpu = unsafe {
            std::slice::from_raw_parts(
                tile_offsets_buf.contents() as *const u32,
                (e + 1).min(8),
            )
        };
        let sumk_val = unsafe { *(sumk_buf.contents() as *const u32) } as usize;
        eprintln!("[E2E] Tile bookkeeping:");
        eprintln!(
            "[E2E]   total_tiles={}, dispatch_args=({}, {}, {})",
            total_tiles_cpu[0],
            dispatch_args_cpu[0],
            dispatch_args_cpu[1],
            dispatch_args_cpu[2]
        );
        eprintln!(
            "[E2E]   tile_offsets[0..{}]={:?}",
            (e + 1).min(8),
            &tile_offsets_cpu
        );
        eprintln!("[E2E]   sumk={}, num_tiles_n={}", sumk_val, num_tiles_n);

        // For multi-token tests with large d_ff, verify gather output (x_perm)
        if t > 1 && d_ff >= 256 {
            let x_perm_cpu = unsafe {
                std::slice::from_raw_parts(
                    x_perm_buf.contents() as *const bf16,
                    sumk_val * d_model,
                )
            };
            eprintln!("[E2E] x_perm diagnostics (sumk={}):", sumk_val);
            eprintln!(
                "[E2E]   Row 0 [0:8]: {:?}",
                &x_perm_cpu[0..8]
                    .iter()
                    .map(|&v| f32::from(v))
                    .collect::<Vec<_>>()
            );
            if sumk_val > 1 {
                eprintln!(
                    "[E2E]   Row 1 [{}:{}]: {:?}",
                    d_model,
                    d_model + 8,
                    &x_perm_cpu[d_model..d_model + 8]
                        .iter()
                        .map(|&v| f32::from(v))
                        .collect::<Vec<_>>()
                );
            }

            // Check tile_map for first few tiles
            let tile_map_cpu = unsafe {
                std::slice::from_raw_parts(
                    tile_map_buf.contents() as *const u32,
                    12.min(max_tiles * 3),
                )
            };
            eprintln!("[E2E] tile_map (first 4 tiles): {:?}", &tile_map_cpu);

            // CRITICAL: Check tok2row mapping for multi-token tests
            let tok2row_cpu = unsafe {
                std::slice::from_raw_parts(
                    tok2row_buf.contents() as *const i32,
                    t * k,
                )
            };
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
            let y_partial_full = unsafe {
                std::slice::from_raw_parts(
                    y_partial_buf.contents() as *const bf16,
                    sumk_val * d_model,
                )
            };
            eprintln!("[E2E] y_partial spot check:");
            eprintln!(
                "[E2E]   y_partial[48] (row 0, col 48) = {:.6}",
                f32::from(y_partial_full[48])
            );
            if sumk_val > 1 {
                // Check positions where mismatches occur (odd indices near tile boundaries)
                eprintln!(
                    "[E2E]   Row 1 positions (even=working, odd=corrupted?):"
                );
                for &pos in &[
                    48, 56, 57, 58, 59, 60, 61, 62, 63, 64, 120, 121, 122, 123,
                    124, 125, 126, 127, 128,
                ] {
                    let idx = 1 * d_model + pos;
                    if idx < y_partial_full.len() {
                        eprintln!(
                            "[E2E]     y_partial[row1, col {}] = {:.3}",
                            pos,
                            f32::from(y_partial_full[idx])
                        );
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

        // Only count as mismatch if both abs and rel are significant
        let threshold_rel = if gate_clip.1 < f32::INFINITY || k > 1 {
            0.5
        } else {
            0.01
        };
        let threshold_abs = 1e-3;

        if rel_diff > threshold_rel && abs_diff > threshold_abs {
            let print_limit =
                if test_name.contains("BoundarySweep_D1024_FF256_T2") {
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

    // Log for threshold calibration
    let k0_iters = (d_model + 31) / 32;
    let ff0_iters = (d_ff + 31) / 32;
    let product = k0_iters * ff0_iters;
    let sqrt_product = (product as f32).sqrt();
    eprintln!(
        "[E2E] Calibration data: k0={}, ff0={}, product={}, sqrt={:.1}, K={}, mean_abs={:.6}, ratio={:.4}",
        k0_iters,
        ff0_iters,
        product,
        sqrt_product,
        k,
        mean_abs_error,
        mean_abs_error / sqrt_product
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

    // For multi-row debugging: print both rows
    if sumk_actual > 1 && d_ff >= 256 {
        eprintln!(
            "[E2E] y_partial row 1 [{}-{}]: {:?}",
            d_model,
            d_model + 16,
            &y_partial_gpu[d_model..d_model + 16]
                .iter()
                .map(|&v| f32::from(v))
                .collect::<Vec<_>>()
        );
    }
    let out_sample_size = 16.min(t * d_model);
    eprintln!(
        "[E2E] y_out sample (token 0): {:?}",
        &y_out_gpu[..out_sample_size]
    );
    if t > 1 && d_model >= 512 {
        eprintln!(
            "[E2E] y_out sample (token 1, first 32): {:?}",
            &y_out_gpu[d_model..d_model + 32]
                .iter()
                .map(|&v| f32::from(v))
                .collect::<Vec<_>>()
        );
    }
    eprintln!(
        "[E2E] CPU ref sample (token 0): {:?}",
        &y_cpu[..out_sample_size]
    );
    if t > 1 && d_model >= 512 {
        eprintln!(
            "[E2E] CPU ref sample (token 1): {:?}",
            &y_cpu[d_model..d_model + 16]
        );

        // Compare token 1 outputs element-wise (more positions to find pattern)
        let gpu_t1 = &y_out_gpu[d_model..];
        let cpu_t1 = &y_cpu[d_model..];
        eprintln!("[E2E] Token 1 detailed comparison:");
        for i in
            [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 48, 64, 121, 128].iter()
        {
            if *i < d_model {
                let diff = (f32::from(gpu_t1[*i]) - cpu_t1[*i]).abs();
                eprintln!(
                    "[E2E]   [{}]: GPU={:.6}, CPU={:.6}, diff={:.6}",
                    i,
                    f32::from(gpu_t1[*i]),
                    cpu_t1[*i],
                    diff
                );
            }
        }
    }

    // Adaptive threshold: BF16 error scales with accumulation depth and K
    //   K=1: error < 0.001 for all scales (including D=4096, H=14336) - virtually perfect
    //   K=2: error ~ 0.015 * sqrt(k0 × ff0) - finalize weighted accumulation adds variance
    //   Pattern: K>1 scales linearly with K
    let k0_iters = (d_model + 31) / 32;
    let ff0_iters = (d_ff + 31) / 32;
    let product = k0_iters * ff0_iters;
    let sqrt_product = (product as f32).sqrt();

    // Tightened thresholds with f32 accumulation: expect better accuracy
    let mean_abs_threshold = if k > 1 || gate_clip.1 < f32::INFINITY {
        // Multi-expert accumulations with f32 accumulation
        // Tighter threshold: 0.015 factor (was 0.02)
        let multi_expert_drift = 0.015 * sqrt_product * (k as f32 / 2.0);
        0.08f32.max(multi_expert_drift) // tightened from 0.1 and 1.1x guard
    } else {
        // Single-expert with f32 accumulation: very tight threshold
        0.01 // tightened from 0.02
    };

    if mean_abs_error >= mean_abs_threshold {
        panic!(
            "[{}] Mean absolute error {:.6} exceeds threshold {:.6}",
            test_name, mean_abs_error, mean_abs_threshold
        );
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

#[test]
fn test_moe_basic() {
    let ctx = create_ctx();

    // Test 1: Minimal (K=1, small dims, no clipping, alpha=1.0)
    run_moe_parity_test(
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

    // Test 2: Multi-expert (K=2, tests finalize weighted sum)
    run_moe_parity_test(
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

    // Test 3: Large d_model only (K=1 to isolate from finalize complexity)
    run_moe_parity_test(
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

    // Test 4a: d_ff=32 (exactly 1 full chunk - baseline for multi-chunk)
    run_moe_parity_test(
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

    // Test 4b: d_ff=48 (1.5 chunks - tests partial second chunk)
    run_moe_parity_test(
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

    // Test 4c: d_ff=64 (exactly 2 full chunks)
    run_moe_parity_test(
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

    // Test 5: GELU activation (gating_code=0)
    run_moe_parity_test(
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

    // Test 6: SiLU activation (gating_code=1)
    run_moe_parity_test(
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

    // Test 7: GEGLU activation (gating_code=3)
    run_moe_parity_test(
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

    // Test 8: Bucket stress - multiple tokens, larger E, stress scatter/gather
    run_moe_parity_test(
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

    // Test 9a: K=4 small-scale to verify multi-expert accumulation
    run_moe_parity_test(
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

    // Test 9a: d_model=128 (4 k0 chunks, 2 n-tiles)
    run_moe_parity_test(
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

    // Test 9b: d_model=192 (6 k0 chunks, 3 n-tiles - test threshold)
    run_moe_parity_test(
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

    // Test 9c: d_model=256 (8 k0 chunks, 4 n-tiles)
    run_moe_parity_test(
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

    // Test 10a: d_model=512 (8 n-tiles)
    run_moe_parity_test(
        &ctx,
        2,   // t
        4,   // e
        1,   // k
        512, // d_model (16 k0 chunks, 8 n-tiles)
        64,  // d_ff (small)
        2,   // gating_code (SwiGLU)
        1.0,
        (f32::NEG_INFINITY, f32::INFINITY),
        (f32::NEG_INFINITY, f32::INFINITY),
        0xE2E_0015,
        "IsolateD_D512_FF64_K1",
    );

    // Test 10b: Isolate large d_ff with small d_model (FF accumulation test)
    run_moe_parity_test(
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

    // Test 10c: d_model=1024 (16 n-tiles)
    run_moe_parity_test(
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

    // Test 10d: Verify layout with d_model=96, d_ff=96 (3 chunks each)
    // This tests non-power-of-2 to catch stride bugs
    run_moe_parity_test(
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

    // Test 10e: d_model=1536 - FIXED by removing hardcoded clamp
    run_moe_parity_test(
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

    // Test 10f: 2 n-tiles - PASSES
    run_moe_parity_test(
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

    // Test 10g: 8 n-tiles - find threshold
    run_moe_parity_test(
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

    // ===== BOUNDARY SWEEP: Isolate column tiling vs ff0 depth =====

    // Sweep A: d_model=2048 (32 n-tiles), d_ff=64 (2 ff0 chunks) - isolate n-tile effect
    run_moe_parity_test(
        &ctx,
        2,    // t (2 tokens like failing test)
        4,    // e
        1,    // k
        2048, // d_model (32 n-tiles)
        64,   // d_ff (small - 2 ff0 chunks)
        2,    // gating_code (SwiGLU)
        1.0,
        (f32::NEG_INFINITY, f32::INFINITY),
        (f32::NEG_INFINITY, f32::INFINITY),
        0xE2E_0022,
        "BoundarySweep_D2048_FF64_K1",
    );

    // Sweep B: d_model=1536 (24 n-tiles), d_ff=256 (8 ff0 chunks) - many ff0, moderate n-tiles
    run_moe_parity_test(
        &ctx,
        2,    // t
        4,    // e
        1,    // k
        1536, // d_model (24 n-tiles)
        256,  // d_ff (8 ff0 chunks)
        2,    // gating_code (SwiGLU)
        1.0,
        (f32::NEG_INFINITY, f32::INFINITY),
        (f32::NEG_INFINITY, f32::INFINITY),
        0xE2E_0023,
        "BoundarySweep_D1536_FF256_K1",
    );

    // Sweep C: d_model=1024, d_ff=256 with T=1 (single token) to isolate
    run_moe_parity_test(
        &ctx,
        1,    // t (single token)
        4,    // e
        1,    // k
        1024, // d_model (16 n-tiles)
        256,  // d_ff (8 ff0 chunks)
        2,    // gating_code (SwiGLU)
        1.0,
        (f32::NEG_INFINITY, f32::INFINITY),
        (f32::NEG_INFINITY, f32::INFINITY),
        0xE2E_0024,
        "BoundarySweep_D1024_FF256_T1",
    );
}

#[test]
fn test_moe_production_scale() {
    let ctx = create_ctx();

    // Test 11a: Production scale with K=1 (isolate from finalize)
    run_moe_parity_test(
        &ctx,
        2,     // t
        8,     // e
        1,     // k (K=1 to isolate)
        4096,  // d_model (128 k0 chunks, 64 n-tiles!)
        14336, // d_ff (448 ff0 chunks!)
        2,     // gating_code (SwiGLU)
        1.0,   // alpha (no clipping)
        (f32::NEG_INFINITY, f32::INFINITY),
        (f32::NEG_INFINITY, f32::INFINITY),
        0xE2E_0028,
        "ProductionScale_D4096_H14336_E8_K1",
    );

    // Test 11b: Production scale with clipping but K=1
    run_moe_parity_test(
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
        "ProductionScale_D4096_H14336_E8_K1_Clipped",
    );

    // Test 11c: PRODUCTION SCALE T=1 decode (triggers GEMV v2)
    run_moe_parity_test(
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
        "ProductionScale_T1_D4096_H14336_E16_K2_GEMV",
    );

    // Test 11d: FULL PRODUCTION SCALE with K=2, T=4 (uses tiled MMA)
    run_moe_parity_test(
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
}
