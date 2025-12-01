//! Unit correctness tests for MoE Expert kernels (both 1-pass and 2-pass variants)
//!
//! Tests verify:
//! - Decode (suffix_length=1) with 2-pass tiled implementation
//! - Prefill (suffix_length>1) with 2-pass tiled implementation
//! - Intermediate buffer correctness (row maps, tiles, dispatch args)
//! - Numerical correctness against CPU reference

#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::bf16;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    KernelDataType,
    kernel::moe::{
        MoeExpertsTwoPassArguments, MoeExpertsTwoPassDecodeKernel,
        MoeExpertsTwoPassPrefillKernel,
        MoeExpertsSingleDecodeArguments, MoeExpertsSingleDecodeKernel,
    },
};

#[path = "moe_test_utils.rs"]
mod test_utils;
use test_utils::{
    alloc_buffer, alloc_buffer_with_data, assert_bf16_close,
    cpu_tile_counts, cpu_tile_scan, create_ctx,
};

/// Build expert segment offsets - distributes sum_k rows across e experts
fn build_offsets(
    e: usize,
    sum_k: usize,
) -> Vec<u32> {
    let mut offsets = vec![0u32; e + 1];
    let base_rows = sum_k / e;
    let extra = sum_k % e;

    for i in 0..=e {
        offsets[i] = (base_rows * i + extra.min(i)) as u32;
    }
    offsets
}

/// Build row-expert map: maps each row to its expert index
fn build_row_expert_map(
    offsets: &[u32],
    sum_k: usize,
) -> Vec<u32> {
    let e = offsets.len() - 1;
    let mut row_map = vec![0u32; sum_k];

    for expert in 0..e {
        let start = offsets[expert] as usize;
        let end = offsets[expert + 1] as usize;
        for row in start..end {
            row_map[row] = expert as u32;
        }
    }
    row_map
}

/// CPU reference for full expert computation matching GPU implementation
///
/// Supports 4 gating modes:
/// - 0: GELU(up)
/// - 1: SiLU(up)
/// - 2: SwiGLU = SiLU(gate) * up
/// - 3: GEGLU = GELU(gate) * up
fn cpu_expert_ffn_full(
    x_perm: &[bf16],      // [sum_k, d_model]
    offsets: &[u32],      // [E+1]
    w13: &[bf16],         // [E, 2*d_ff, d_model] transposed
    w2: &[bf16],          // [E, d_model, d_ff] transposed
    up_biases: &[bf16],   // [E, 2*d_ff]
    down_biases: &[bf16], // [E, d_model]
    d_model: usize,
    d_ff: usize,
    gating_code: u32,
    silu_alpha: f32,
) -> (Vec<f32>, Vec<bf16>) {
    let e = offsets.len() - 1;
    let sum_k = x_perm.len() / d_model;
    let mut hidden_full = vec![0.0f32; sum_k * d_ff];
    let mut output = vec![bf16::from_f32(0.0); sum_k * d_model];

    // Helper functions matching Metal kernels
    let gelu_approx = |x: f32| -> f32 {
        const K0: f32 = 0.7978845608;
        const K1: f32 = 0.044715;
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
    };

    let silu = |x: f32, alpha: f32| -> f32 { x / (1.0 + (-alpha * x).exp()) };

    for expert in 0..e {
        let row_start = offsets[expert] as usize;
        let row_end = offsets[expert + 1] as usize;
        let expert_rows = row_end - row_start;

        if expert_rows == 0 {
            continue;
        }

        let w1_offset = expert * 2 * d_ff * d_model;
        let w2_offset = expert * d_model * d_ff;

        // FC1: [expert_rows, d_model] @ [d_model, 2*d_ff] -> [expert_rows, 2*d_ff]
        // Use f32 for hidden to match GPU which stores hidden in f32 between passes
        let mut hidden = vec![0.0f32; expert_rows * 2 * d_ff];
        for row in 0..expert_rows {
            for ff in 0..(2 * d_ff) {
                let mut acc = f32::from(up_biases[expert * 2 * d_ff + ff]);
                for dm in 0..d_model {
                    let x_val =
                        f32::from(x_perm[(row_start + row) * d_model + dm]);
                    // w13 transposed: [2*d_ff, d_model]
                    let w_val = f32::from(w13[w1_offset + ff * d_model + dm]);
                    acc += x_val * w_val;
                }
                hidden[row * 2 * d_ff + ff] = acc;
            }
        }

        // Apply activation based on gating_code
        for row in 0..expert_rows {
            for ff in 0..d_ff {
                let up = hidden[row * 2 * d_ff + ff];

                let result = if gating_code <= 1 {
                    // GELU(up) or SiLU(up) - no gating
                    if gating_code == 0 {
                        gelu_approx(up)
                    } else {
                        silu(up, silu_alpha)
                    }
                } else {
                    // SwiGLU or GEGLU - gate * up
                    let gate = hidden[row * 2 * d_ff + d_ff + ff];
                    let activated_gate = if gating_code == 2 {
                        silu(gate, silu_alpha)
                    } else {
                        gelu_approx(gate)
                    };
                    activated_gate * up
                };

                hidden[row * 2 * d_ff + ff] = result;
                hidden_full[(row_start + row) * d_ff + ff] = result;
            }
        }

        // FC2: [expert_rows, d_ff] @ [d_ff, d_model] -> [expert_rows, d_model]
        for row in 0..expert_rows {
            for dm in 0..d_model {
                let mut acc = f32::from(down_biases[expert * d_model + dm]);
                for ff in 0..d_ff {
                    let h_val = hidden[row * 2 * d_ff + ff];
                    // w2 transposed: [d_model, d_ff]
                    let w_val = f32::from(w2[w2_offset + dm * d_ff + ff]);
                    acc += h_val * w_val;
                }
                output[(row_start + row) * d_model + dm] = bf16::from_f32(acc);
            }
        }
    }

    (hidden_full, output)
}

/// CPU reference for single-token MoE decode with weighted expert sum
///
/// Computes: y = Σ_k prob[k] * (activation(x @ W13[expert_k]) @ W2[expert_k] + bias)
fn cpu_moe_single_token(
    x: &[bf16],
    topk_ids: &[i32],
    topk_probs: &[bf16],
    w13_all: &[bf16],
    w2_all: &[bf16],
    up_biases: &[bf16],
    down_biases: &[bf16],
    d_model: usize,
    d_ff: usize,
    gating_code: u32,
) -> Vec<bf16> {
    let k = topk_ids.len();

    // Hidden buffer [K, d_ff]
    let mut hidden = vec![0.0f32; k * d_ff];
    // Final output [d_model]
    let mut y = vec![bf16::from_f32(0.0); d_model];

    let gelu_approx = |x: f32| -> f32 {
        const K0: f32 = 0.7978845608;
        const K1: f32 = 0.044715;
        if x > 10.0 { return x; }
        if x < -10.0 { return 0.0; }
        let x3 = x * x * x;
        let inner = x + K1 * x3;
        let tanh_arg = (K0 * inner).clamp(-10.0, 10.0);
        0.5 * x * (1.0 + tanh_arg.tanh())
    };

    let silu = |x: f32| -> f32 { x / (1.0 + (-x).exp()) };

    // Pass A: x @ W13[expert] -> hidden[k]
    for k_idx in 0..k {
        let expert_id = topk_ids[k_idx];
        if expert_id < 0 { continue; }
        let expert = expert_id as usize;

        let w13_base = expert * 2 * d_ff * d_model;
        let bias_base = expert * 2 * d_ff;

        for h in 0..d_ff {
            // Up projection
            let mut acc_up = f32::from(up_biases[bias_base + h]);
            for d in 0..d_model {
                let x_val = f32::from(x[d]);
                let w_val = f32::from(w13_all[w13_base + h * d_model + d]);
                acc_up += x_val * w_val;
            }

            // Gate projection (for SwiGLU/GEGLU)
            let activated = if gating_code <= 1 {
                // GELU or SiLU on up
                if gating_code == 0 { gelu_approx(acc_up) } else { silu(acc_up) }
            } else {
                // SwiGLU or GEGLU
                let mut acc_gate = f32::from(up_biases[bias_base + d_ff + h]);
                for d in 0..d_model {
                    let x_val = f32::from(x[d]);
                    let w_val = f32::from(w13_all[w13_base + (d_ff + h) * d_model + d]);
                    acc_gate += x_val * w_val;
                }
                let gate_act = if gating_code == 2 { silu(acc_gate) } else { gelu_approx(acc_gate) };
                gate_act * acc_up
            };

            hidden[k_idx * d_ff + h] = activated;
        }
    }

    // Pass B: hidden @ W2 -> y (with weighted sum)
    for d in 0..d_model {
        let mut final_acc = 0.0f32;

        for k_idx in 0..k {
            let expert_id = topk_ids[k_idx];
            if expert_id < 0 { continue; }
            let expert = expert_id as usize;
            let prob = f32::from(topk_probs[k_idx]);

            let w2_base = expert * d_model * d_ff;
            let bias_base = expert * d_model;

            let mut acc = f32::from(down_biases[bias_base + d]);
            for h in 0..d_ff {
                let h_val = hidden[k_idx * d_ff + h];
                let w_val = f32::from(w2_all[w2_base + d * d_ff + h]);
                acc += h_val * w_val;
            }
            final_acc += prob * acc;
        }

        y[d] = bf16::from_f32(final_acc);
    }

    y
}

#[test]
fn test_two_pass_decode_correctness() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xDEC0DE42);

    // Realistic decode test case: single token, K=2
    let t = 1;
    let k = 2;
    let sum_k = t * k;
    let d_model = 512;
    let d_ff = 2048;
    let e = 8;

    eprintln!(
        "[2-pass decode] T={}, K={}, sum_k={}, d_model={}, d_ff={}, E={}",
        t, k, sum_k, d_model, d_ff, e
    );

    // Generate test data
    let x_perm: Vec<bf16> = (0..sum_k * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
        .collect();

    let offsets = build_offsets(e, sum_k);
    let row_expert_map = build_row_expert_map(&offsets, sum_k);

    // Generate weights (W13 in GPU transposed layout [E, 2*d_ff, d_model])
    let w13: Vec<bf16> = (0..e * 2 * d_ff * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.05..0.05)))
        .collect();
    let w2: Vec<bf16> = (0..e * d_ff * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.05..0.05)))
        .collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.random_range(-0.01..0.01)))
        .collect();
    let down_biases: Vec<bf16> = (0..e * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.01..0.01)))
        .collect();

    // CPU reference (gating_code=0 means GELU)
    let (_hidden_expected, expected) = cpu_expert_ffn_full(
        &x_perm,
        &offsets,
        &w13,
        &w2,
        &up_biases,
        &down_biases,
        d_model,
        d_ff,
        0,
        1.0,
    );

    // Prepare GPU buffers
    let x_perm_buf = alloc_buffer_with_data(&ctx, &x_perm);
    let offsets_buf = alloc_buffer_with_data(&ctx, &offsets);
    let row_expert_map_buf = alloc_buffer_with_data(&ctx, &row_expert_map);
    let w13_buf = alloc_buffer_with_data(&ctx, &w13);
    let w2_buf = alloc_buffer_with_data(&ctx, &w2);
    let up_biases_buf = alloc_buffer_with_data(&ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data(&ctx, &down_biases);

    // Intermediate buffers - hidden is f32 for activation precision
    let hidden_buf = alloc_buffer::<f32>(&ctx, sum_k * d_ff);
    let output_buf = alloc_buffer::<bf16>(&ctx, sum_k * d_model);

    // Tile infrastructure
    let h_blocks_decode = (d_ff + 3) / 4;
    let max_total_tiles = sum_k * h_blocks_decode;

    let tile_counts_buf = alloc_buffer::<u32>(&ctx, e);
    let tile_offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let total_tiles_buf = alloc_buffer::<u32>(&ctx, 8);
    let tile_map_buf = alloc_buffer::<u32>(&ctx, max_total_tiles * 3);
    let dispatch_args_buf = alloc_buffer::<u32>(&ctx, 3);

    // Execute 2-pass decode kernel
    let experts_kernel = MoeExpertsTwoPassDecodeKernel::new(&ctx)
        .expect("MoeExpertsTwoPassDecodeKernel::new");
    let cb = ctx.command_queue.new_command_buffer();

    const K_TILE: usize = 64;
    let num_tiles_k = ((d_ff + K_TILE - 1) / K_TILE) as u32;

    experts_kernel
        .encode(
            &cb,
            MoeExpertsTwoPassArguments {
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
                num_tiles_k,
                gating_code: 0, // GELU activation
                gate_clip_min: -1e6,
                gate_clip_max: 1e6,
                up_clip_min: -1e6,
                up_clip_max: 1e6,
                silu_alpha: 1.0,
                data_type: KernelDataType::BFloat16,
            },
        )
        .expect("encode_two_pass_decode");

    cb.commit();
    cb.wait_until_completed();

    // Read GPU output
    let output_gpu = unsafe {
        std::slice::from_raw_parts(
            output_buf.contents() as *const bf16,
            sum_k * d_model,
        )
    };

    // Compute error metrics
    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut max_abs_idx = 0;
    let mut max_rel_idx = 0;

    for (i, (&gpu_val, &cpu_val)) in
        output_gpu.iter().zip(expected.iter()).enumerate()
    {
        let gpu_f = f32::from(gpu_val);
        let cpu_f = f32::from(cpu_val);
        let abs_error = (gpu_f - cpu_f).abs();
        let rel_error = if cpu_f.abs() > 1e-6 {
            abs_error / cpu_f.abs()
        } else {
            0.0
        };

        if abs_error > max_abs_error {
            max_abs_error = abs_error;
            max_abs_idx = i;
        }
        if rel_error > max_rel_error {
            max_rel_error = rel_error;
            max_rel_idx = i;
        }
    }

    eprintln!(
        "[2-pass decode] Max absolute error: {:.6} at index {} (GPU={:.6}, CPU={:.6})",
        max_abs_error,
        max_abs_idx,
        f32::from(output_gpu[max_abs_idx]),
        f32::from(expected[max_abs_idx])
    );
    eprintln!(
        "[2-pass decode] Max relative error: {:.6} ({:.2}%) at index {} (GPU={:.6}, CPU={:.6})",
        max_rel_error,
        max_rel_error * 100.0,
        max_rel_idx,
        f32::from(output_gpu[max_rel_idx]),
        f32::from(expected[max_rel_idx])
    );

    let tolerance = 0.01;

    assert_bf16_close(output_gpu, &expected, tolerance, "2-pass decode output");

    eprintln!("[2-pass decode] ✓ PASSED (tolerance={:.4})", tolerance);
}

#[test]
fn test_two_pass_decode_multi_token() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xDEC0DE43);

    // Multi-token decode: T=4 tokens, K=2 (tests indirect decode path with T>1)
    let t = 4;
    let k = 2;
    let sum_k = t * k;
    let d_model = 512;
    let d_ff = 2048;
    let e = 8;

    eprintln!(
        "[2-pass decode multi-token] T={}, K={}, sum_k={}, d_model={}, d_ff={}, E={}",
        t, k, sum_k, d_model, d_ff, e
    );

    // Generate test data
    let x_perm: Vec<bf16> = (0..sum_k * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
        .collect();

    let offsets = build_offsets(e, sum_k);
    let row_expert_map = build_row_expert_map(&offsets, sum_k);

    let w13: Vec<bf16> = (0..e * 2 * d_ff * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.05..0.05)))
        .collect();
    let w2: Vec<bf16> = (0..e * d_ff * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.05..0.05)))
        .collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.random_range(-0.01..0.01)))
        .collect();
    let down_biases: Vec<bf16> = (0..e * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.01..0.01)))
        .collect();

    // CPU reference (gating_code=0 means GELU)
    let (_hidden_expected, expected) = cpu_expert_ffn_full(
        &x_perm,
        &offsets,
        &w13,
        &w2,
        &up_biases,
        &down_biases,
        d_model,
        d_ff,
        0,
        1.0,
    );

    // Prepare GPU buffers
    let x_perm_buf = alloc_buffer_with_data(&ctx, &x_perm);
    let offsets_buf = alloc_buffer_with_data(&ctx, &offsets);
    let row_expert_map_buf = alloc_buffer_with_data(&ctx, &row_expert_map);
    let w13_buf = alloc_buffer_with_data(&ctx, &w13);
    let w2_buf = alloc_buffer_with_data(&ctx, &w2);
    let up_biases_buf = alloc_buffer_with_data(&ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data(&ctx, &down_biases);

    let hidden_buf = alloc_buffer::<f32>(&ctx, sum_k * d_ff);
    let output_buf = alloc_buffer::<bf16>(&ctx, sum_k * d_model);

    // Tile infrastructure
    let h_blocks_decode = (d_ff + 3) / 4;
    let max_total_tiles = sum_k * h_blocks_decode;

    let tile_counts_buf = alloc_buffer::<u32>(&ctx, e);
    let tile_offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let total_tiles_buf = alloc_buffer::<u32>(&ctx, 8);
    let tile_map_buf = alloc_buffer::<u32>(&ctx, max_total_tiles * 3);
    let dispatch_args_buf = alloc_buffer::<u32>(&ctx, 3);

    // Execute 2-pass decode kernel
    let experts_kernel = MoeExpertsTwoPassDecodeKernel::new(&ctx)
        .expect("MoeExpertsTwoPassDecodeKernel::new");
    let cb = ctx.command_queue.new_command_buffer();

    const K_TILE: usize = 64;
    let num_tiles_k = ((d_ff + K_TILE - 1) / K_TILE) as u32;

    experts_kernel
        .encode(
            &cb,
            MoeExpertsTwoPassArguments {
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
                num_tiles_k,
                gating_code: 0,
                gate_clip_min: -1e6,
                gate_clip_max: 1e6,
                up_clip_min: -1e6,
                up_clip_max: 1e6,
                silu_alpha: 1.0,
                data_type: KernelDataType::BFloat16,
            },
        )
        .expect("encode_two_pass_decode");

    cb.commit();
    cb.wait_until_completed();

    let output_gpu = unsafe {
        std::slice::from_raw_parts(
            output_buf.contents() as *const bf16,
            sum_k * d_model,
        )
    };

    // Compute error metrics
    let mut max_abs_error = 0.0f32;
    let mut sum_abs_error = 0.0f32;
    let mut max_abs_idx = 0;

    for (i, (&gpu_val, &cpu_val)) in
        output_gpu.iter().zip(expected.iter()).enumerate()
    {
        let gpu_f = f32::from(gpu_val);
        let cpu_f = f32::from(cpu_val);
        let abs_error = (gpu_f - cpu_f).abs();
        sum_abs_error += abs_error;

        if abs_error > max_abs_error {
            max_abs_error = abs_error;
            max_abs_idx = i;
        }
    }

    let mean_abs_error = sum_abs_error / (sum_k * d_model) as f32;

    eprintln!(
        "[2-pass decode multi-token] Max error: {:.6} at index {} (GPU={:.6}, CPU={:.6}), Mean error: {:.6}",
        max_abs_error,
        max_abs_idx,
        f32::from(output_gpu[max_abs_idx]),
        f32::from(expected[max_abs_idx]),
        mean_abs_error
    );

    let tolerance = 0.01;
    assert_bf16_close(output_gpu, &expected, tolerance, "2-pass decode multi-token output");

    eprintln!("[2-pass decode multi-token] ✓ PASSED (tolerance={:.4})", tolerance);
}

#[test]
fn test_two_pass_prefill_correctness() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xFE1111);

    // Realistic prefill case: 8 tokens, K=2
    let t = 8;
    let k = 2;
    let sum_k = t * k;
    let d_model = 512;
    let d_ff = 2048;
    let e = 8;

    eprintln!(
        "[2-pass prefill] T={}, K={}, sum_k={}, d_model={}, d_ff={}, E={}",
        t, k, sum_k, d_model, d_ff, e
    );

    // Generate test data
    let x_perm: Vec<bf16> = (0..sum_k * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
        .collect();

    let offsets = build_offsets(e, sum_k);
    let row_expert_map = build_row_expert_map(&offsets, sum_k);

    // Generate weights
    let w13: Vec<bf16> = (0..e * 2 * d_ff * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.05..0.05)))
        .collect();
    let w2: Vec<bf16> = (0..e * d_ff * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.05..0.05)))
        .collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.random_range(-0.01..0.01)))
        .collect();
    let down_biases: Vec<bf16> = (0..e * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.01..0.01)))
        .collect();

    // CPU reference (gating_code=0 means GELU)
    let (hidden_expected, expected) = cpu_expert_ffn_full(
        &x_perm,
        &offsets,
        &w13,
        &w2,
        &up_biases,
        &down_biases,
        d_model,
        d_ff,
        0,
        1.0,
    );

    // Prepare GPU buffers
    let x_perm_buf = alloc_buffer_with_data(&ctx, &x_perm);
    let offsets_buf = alloc_buffer_with_data(&ctx, &offsets);
    let row_expert_map_buf = alloc_buffer_with_data(&ctx, &row_expert_map);
    let w13_buf = alloc_buffer_with_data(&ctx, &w13);
    let w2_buf = alloc_buffer_with_data(&ctx, &w2);
    let up_biases_buf = alloc_buffer_with_data(&ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data(&ctx, &down_biases);

    // Intermediate buffers - hidden is f32 for activation precision
    let hidden_buf = alloc_buffer::<f32>(&ctx, sum_k * d_ff);
    let output_buf = alloc_buffer::<bf16>(&ctx, sum_k * d_model);

    // Tile infrastructure
    let h_blocks_decode = (d_ff + 3) / 4;
    let max_total_tiles = sum_k * h_blocks_decode;

    let tile_counts_buf = alloc_buffer::<u32>(&ctx, e);
    let tile_offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let total_tiles_buf = alloc_buffer::<u32>(&ctx, 8);
    let tile_map_buf = alloc_buffer::<u32>(&ctx, max_total_tiles * 3);
    let dispatch_args_buf = alloc_buffer::<u32>(&ctx, 3);

    // Execute 2-pass prefill kernel
    let experts_kernel = MoeExpertsTwoPassPrefillKernel::new(&ctx)
        .expect("MoeExpertsTwoPassPrefillKernel::new");
    let cb = ctx.command_queue.new_command_buffer();

    const K_TILE: usize = 64;
    let num_tiles_k = ((d_ff + K_TILE - 1) / K_TILE) as u32;

    experts_kernel
        .encode(
            &cb,
            MoeExpertsTwoPassArguments {
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
                num_tiles_k,
                gating_code: 0, // GELU activation
                gate_clip_min: -1e6,
                gate_clip_max: 1e6,
                up_clip_min: -1e6,
                up_clip_max: 1e6,
                silu_alpha: 1.0,
                data_type: KernelDataType::BFloat16,
            },
        )
        .expect("encode_two_pass_prefill");

    cb.commit();
    cb.wait_until_completed();

    // Read GPU output
    let output_gpu = unsafe {
        std::slice::from_raw_parts(
            output_buf.contents() as *const bf16,
            sum_k * d_model,
        )
    };

    // Compute error metrics
    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut max_abs_idx = 0;
    let mut max_rel_idx = 0;

    for (i, (&gpu_val, &cpu_val)) in
        output_gpu.iter().zip(expected.iter()).enumerate()
    {
        let gpu_f = f32::from(gpu_val);
        let cpu_f = f32::from(cpu_val);
        let abs_error = (gpu_f - cpu_f).abs();
        let rel_error = if cpu_f.abs() > 1e-6 {
            abs_error / cpu_f.abs()
        } else {
            0.0
        };

        if abs_error > max_abs_error {
            max_abs_error = abs_error;
            max_abs_idx = i;
        }
        if rel_error > max_rel_error {
            max_rel_error = rel_error;
            max_rel_idx = i;
        }
    }

    eprintln!(
        "[2-pass prefill] Max absolute error: {:.6} at index {} (GPU={:.6}, CPU={:.6})",
        max_abs_error,
        max_abs_idx,
        f32::from(output_gpu[max_abs_idx]),
        f32::from(expected[max_abs_idx])
    );
    eprintln!(
        "[2-pass prefill] Max relative error: {:.6} ({:.2}%) at index {} (GPU={:.6}, CPU={:.6})",
        max_rel_error,
        max_rel_error * 100.0,
        max_rel_idx,
        f32::from(output_gpu[max_rel_idx]),
        f32::from(expected[max_rel_idx])
    );

    // Verify correctness with tight tolerance
    let tolerance = 0.01;

    let hidden_gpu = unsafe {
        std::slice::from_raw_parts(
            hidden_buf.contents() as *const f32,
            sum_k * d_ff,
        )
    };

    let mut hidden_max_diff = 0.0f32;
    let mut hidden_max_idx = 0usize;
    for (idx, (&gpu, &cpu)) in
        hidden_gpu.iter().zip(hidden_expected.iter()).enumerate()
    {
        let diff = (gpu - cpu).abs();
        if diff > hidden_max_diff {
            hidden_max_diff = diff;
            hidden_max_idx = idx;
        }
    }

    if hidden_max_diff > tolerance {
        let row = hidden_max_idx / d_ff;
        let col = hidden_max_idx % d_ff;
        eprintln!(
            "[2-pass prefill] hidden mismatch row={} col={} gpu={} cpu={} diff={}",
            row,
            col,
            hidden_gpu[hidden_max_idx],
            hidden_expected[hidden_max_idx],
            hidden_max_diff
        );

        for offset in 0..8 {
            let idx = row * d_ff + (col + offset).min(d_ff - 1);
            let g = hidden_gpu[idx];
            let c = hidden_expected[idx];
            eprintln!(
                "    col {} -> gpu {:.6} cpu {:.6} diff {:.6}",
                (col + offset).min(d_ff - 1),
                g,
                c,
                (g - c).abs()
            );
        }
    } else {
        eprintln!(
            "[2-pass prefill] hidden max diff {:.6} within tolerance",
            hidden_max_diff
        );
    }

    assert_bf16_close(
        output_gpu,
        &expected,
        tolerance,
        "2-pass prefill output",
    );

    eprintln!("[2-pass prefill] ✓ PASSED (tolerance={:.4})", tolerance);
}

#[test]
fn test_tile_infrastructure() {
    // Test that tile counts/scan/map are computed correctly
    // This is already tested in moe_tiles_test.rs, but we verify here
    // that the tiles match our expectations for expert workload distribution

    let offsets = vec![0, 5, 20, 35, 40];
    const BM: usize = 16;

    let tile_counts = cpu_tile_counts(&offsets, BM);
    let (tile_offsets, total_tiles) = cpu_tile_scan(&tile_counts);

    // Expert 0: 5 rows -> 1 tile
    // Expert 1: 15 rows -> 1 tile
    // Expert 2: 15 rows -> 1 tile
    // Expert 3: 5 rows -> 1 tile
    assert_eq!(tile_counts, vec![1, 1, 1, 1]);
    assert_eq!(tile_offsets, vec![0, 1, 2, 3, 4]);
    assert_eq!(total_tiles, 4);

    eprintln!("[tile infrastructure] ✓ PASSED");
}

#[test]
fn test_fused_single_token_decode() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xF053ED);

    let d_model = 512;
    let d_ff = 2048;
    let e = 8;
    let k = 2;

    eprintln!(
        "[fused single-token] d_model={}, d_ff={}, E={}, K={}",
        d_model, d_ff, e, k
    );

    // Generate test data
    let x: Vec<bf16> = (0..d_model)
        .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
        .collect();

    let topk_ids: Vec<i32> = (0..k).map(|i| (i % e) as i32).collect();

    let topk_probs: Vec<bf16> = {
        let raw: Vec<f32> = (0..k).map(|_| rng.random_range(0.1..1.0)).collect();
        let sum: f32 = raw.iter().sum();
        raw.iter().map(|p| bf16::from_f32(p / sum)).collect()
    };

    let w13_all: Vec<bf16> = (0..e * 2 * d_ff * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.05..0.05)))
        .collect();
    let w2_all: Vec<bf16> = (0..e * d_ff * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.05..0.05)))
        .collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.random_range(-0.01..0.01)))
        .collect();
    let down_biases: Vec<bf16> = (0..e * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.01..0.01)))
        .collect();

    // CPU reference (gating_code=2 for SwiGLU)
    let gating_code = 2u32;
    let y_expected = cpu_moe_single_token(
        &x, &topk_ids, &topk_probs,
        &w13_all, &w2_all, &up_biases, &down_biases,
        d_model, d_ff, gating_code,
    );

    // GPU buffers
    let x_buf = alloc_buffer_with_data(&ctx, &x);
    let topk_ids_buf = alloc_buffer_with_data(&ctx, &topk_ids);
    let topk_probs_buf = alloc_buffer_with_data(&ctx, &topk_probs);
    let w13_buf = alloc_buffer_with_data(&ctx, &w13_all);
    let w2_buf = alloc_buffer_with_data(&ctx, &w2_all);
    let up_biases_buf = alloc_buffer_with_data(&ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data(&ctx, &down_biases);
    let hidden_buf = alloc_buffer::<f32>(&ctx, k * d_ff);
    let y_buf = alloc_buffer::<bf16>(&ctx, d_model);

    // Run fused decode kernel
    let fused_kernel = MoeExpertsSingleDecodeKernel::new(&ctx)
        .expect("MoeExpertsSingleDecodeKernel::new");
    let cb = ctx.command_queue.new_command_buffer();

    fused_kernel
        .encode(
            &cb,
            MoeExpertsSingleDecodeArguments {
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
                gating_code,
                data_type: KernelDataType::BFloat16,
            },
        )
        .expect("fused decode encode");

    cb.commit();
    cb.wait_until_completed();

    // Read GPU output
    let y_gpu = unsafe {
        std::slice::from_raw_parts(y_buf.contents() as *const bf16, d_model)
    };

    // Compute error metrics
    let mut max_abs_error = 0.0f32;
    let mut sum_abs_error = 0.0f32;
    let mut max_idx = 0;

    for (i, (&gpu_val, &cpu_val)) in y_gpu.iter().zip(y_expected.iter()).enumerate() {
        let abs_error = (f32::from(gpu_val) - f32::from(cpu_val)).abs();
        sum_abs_error += abs_error;
        if abs_error > max_abs_error {
            max_abs_error = abs_error;
            max_idx = i;
        }
    }
    let mean_abs_error = sum_abs_error / d_model as f32;

    eprintln!(
        "[fused single-token] Max error: {:.6} at idx {} (GPU={:.6}, CPU={:.6}), Mean error: {:.6}",
        max_abs_error, max_idx,
        f32::from(y_gpu[max_idx]), f32::from(y_expected[max_idx]),
        mean_abs_error
    );

    let tolerance = 0.02;
    assert_bf16_close(y_gpu, &y_expected, tolerance, "fused single-token output");

    eprintln!("[fused single-token] PASSED (tolerance={:.4})", tolerance);
}

#[test]
fn test_fused_single_token_k4() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xE001A);

    let d_model = 512;
    let d_ff = 2048;
    let e = 8;
    let k = 4;

    eprintln!(
        "[fused single-token K=4] d_model={}, d_ff={}, E={}, K={}",
        d_model, d_ff, e, k
    );

    let x: Vec<bf16> = (0..d_model)
        .map(|_| bf16::from_f32(rng.random_range(-1.0..1.0)))
        .collect();

    let topk_ids: Vec<i32> = (0..k).map(|i| ((i * 2) % e) as i32).collect();

    let topk_probs: Vec<bf16> = {
        let raw: Vec<f32> = (0..k).map(|_| rng.random_range(0.1..1.0)).collect();
        let sum: f32 = raw.iter().sum();
        raw.iter().map(|p| bf16::from_f32(p / sum)).collect()
    };

    let w13_all: Vec<bf16> = (0..e * 2 * d_ff * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.05..0.05)))
        .collect();
    let w2_all: Vec<bf16> = (0..e * d_ff * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.05..0.05)))
        .collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff)
        .map(|_| bf16::from_f32(rng.random_range(-0.01..0.01)))
        .collect();
    let down_biases: Vec<bf16> = (0..e * d_model)
        .map(|_| bf16::from_f32(rng.random_range(-0.01..0.01)))
        .collect();

    let gating_code = 2u32;
    let y_expected = cpu_moe_single_token(
        &x, &topk_ids, &topk_probs,
        &w13_all, &w2_all, &up_biases, &down_biases,
        d_model, d_ff, gating_code,
    );

    let x_buf = alloc_buffer_with_data(&ctx, &x);
    let topk_ids_buf = alloc_buffer_with_data(&ctx, &topk_ids);
    let topk_probs_buf = alloc_buffer_with_data(&ctx, &topk_probs);
    let w13_buf = alloc_buffer_with_data(&ctx, &w13_all);
    let w2_buf = alloc_buffer_with_data(&ctx, &w2_all);
    let up_biases_buf = alloc_buffer_with_data(&ctx, &up_biases);
    let down_biases_buf = alloc_buffer_with_data(&ctx, &down_biases);
    let hidden_buf = alloc_buffer::<f32>(&ctx, k * d_ff);
    let y_buf = alloc_buffer::<bf16>(&ctx, d_model);

    let fused_kernel = MoeExpertsSingleDecodeKernel::new(&ctx).expect("fused kernel");
    let cb = ctx.command_queue.new_command_buffer();
    fused_kernel
        .encode(&cb, MoeExpertsSingleDecodeArguments {
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
            gating_code,
            data_type: KernelDataType::BFloat16,
        })
        .expect("fused encode");
    cb.commit();
    cb.wait_until_completed();

    let y_gpu = unsafe {
        std::slice::from_raw_parts(y_buf.contents() as *const bf16, d_model)
    };

    // Compute error metrics
    let mut max_abs_error = 0.0f32;
    let mut sum_abs_error = 0.0f32;
    let mut max_idx = 0;

    for (i, (&gpu_val, &cpu_val)) in y_gpu.iter().zip(y_expected.iter()).enumerate() {
        let abs_error = (f32::from(gpu_val) - f32::from(cpu_val)).abs();
        sum_abs_error += abs_error;
        if abs_error > max_abs_error {
            max_abs_error = abs_error;
            max_idx = i;
        }
    }
    let mean_abs_error = sum_abs_error / d_model as f32;

    eprintln!(
        "[fused single-token K=4] Max error: {:.6} at idx {} (GPU={:.6}, CPU={:.6}), Mean error: {:.6}",
        max_abs_error, max_idx,
        f32::from(y_gpu[max_idx]), f32::from(y_expected[max_idx]),
        mean_abs_error
    );

    let tolerance = 0.02;
    assert_bf16_close(y_gpu, &y_expected, tolerance, "fused single-token K=4 output");

    eprintln!("[fused single-token K=4] PASSED (tolerance={:.4})", tolerance);
}
