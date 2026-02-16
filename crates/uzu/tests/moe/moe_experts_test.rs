//! Unit correctness tests for MoE Expert kernels (both 1-pass and 2-pass variants)
//!
//! Tests verify:
//! - Decode (suffix_length=1) with 2-pass tiled implementation
//! - Prefill (suffix_length>1) with 2-pass tiled implementation
//! - Intermediate buffer correctness (row maps, tiles, dispatch args)
//! - Numerical correctness against CPU reference

#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::bf16;
use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::{
    DataType,
    backends::metal::kernel::moe::{
        MoeExpertsSingleDecodeKernels, MoeExpertsTwoPassArguments, MoeExpertsTwoPassDecodeKernels,
        MoeExpertsTwoPassPrefillKernel,
    },
};

#[path = "moe_test_utils.rs"]
mod test_utils;
use test_utils::{alloc_buffer, alloc_buffer_with_data, assert_bf16_close, cpu_tile_counts, cpu_tile_scan, create_ctx};
use uzu::backends::metal::kernel::moe::MoeExpertsSingleDecodeArguments;

/// Test data for MoE experts
struct MoeTestData {
    x: Vec<bf16>,
    topk_ids: Vec<i32>,
    topk_probs: Vec<bf16>,
    w13: Vec<bf16>,
    w2: Vec<bf16>,
    up_biases: Vec<bf16>,
    down_biases: Vec<bf16>,
}

impl MoeTestData {
    fn generate(
        rng: &mut StdRng,
        t: usize,
        k: usize,
        d_model: usize,
        d_ff: usize,
        e: usize,
    ) -> Self {
        let x: Vec<bf16> = (0..t * d_model).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect();

        let topk_ids: Vec<i32> = (0..t * k).map(|i| ((i / k) + (i % k) * 2) as i32 % e as i32).collect();

        let topk_probs: Vec<bf16> = (0..t)
            .flat_map(|_| {
                let raw: Vec<f32> = (0..k).map(|_| rng.random_range(0.1..1.0)).collect();
                let sum: f32 = raw.iter().sum();
                raw.iter().map(|p| bf16::from_f32(p / sum)).collect::<Vec<_>>()
            })
            .collect();

        let w13: Vec<bf16> =
            (0..e * 2 * d_ff * d_model).map(|_| bf16::from_f32(rng.random_range(-0.05..0.05))).collect();
        let w2: Vec<bf16> = (0..e * d_ff * d_model).map(|_| bf16::from_f32(rng.random_range(-0.05..0.05))).collect();
        let up_biases: Vec<bf16> = (0..e * 2 * d_ff).map(|_| bf16::from_f32(rng.random_range(-0.01..0.01))).collect();
        let down_biases: Vec<bf16> = (0..e * d_model).map(|_| bf16::from_f32(rng.random_range(-0.01..0.01))).collect();

        Self {
            x,
            topk_ids,
            topk_probs,
            w13,
            w2,
            up_biases,
            down_biases,
        }
    }
}

/// Result of scatter operation - buckets rows by expert
struct ScatterResult {
    x_perm: Vec<bf16>,
    offsets: Vec<u32>,
    row_expert_map: Vec<u32>,
    perm_idx: Vec<usize>, // original row -> bucketed row (for gather)
}

fn scatter_by_expert(
    x: &[bf16],
    topk_ids: &[i32],
    t: usize,
    k: usize,
    d_model: usize,
    e: usize,
) -> ScatterResult {
    let sum_k = t * k;

    // Count rows per expert and build offsets
    let mut expert_counts = vec![0usize; e];
    for &eid in topk_ids {
        expert_counts[eid as usize] += 1;
    }
    let mut offsets = vec![0u32; e + 1];
    for i in 0..e {
        offsets[i + 1] = offsets[i] + expert_counts[i] as u32;
    }

    // Build permutation indices
    let mut expert_cursors = vec![0usize; e];
    let mut perm_idx = vec![0usize; sum_k];
    let mut inv_perm = vec![0usize; sum_k];
    for orig_row in 0..sum_k {
        let eid = topk_ids[orig_row] as usize;
        let bucket_pos = offsets[eid] as usize + expert_cursors[eid];
        perm_idx[orig_row] = bucket_pos;
        inv_perm[bucket_pos] = orig_row;
        expert_cursors[eid] += 1;
    }

    // Build x_perm and row_expert_map in bucketed order
    let mut x_perm = vec![bf16::from_f32(0.0); sum_k * d_model];
    let mut row_expert_map = vec![0u32; sum_k];
    for bucket_row in 0..sum_k {
        let orig_row = inv_perm[bucket_row];
        let tok = orig_row / k;
        for d in 0..d_model {
            x_perm[bucket_row * d_model + d] = x[tok * d_model + d];
        }
        row_expert_map[bucket_row] = topk_ids[orig_row] as u32;
    }

    ScatterResult {
        x_perm,
        offsets,
        row_expert_map,
        perm_idx,
    }
}

/// Gather + finalize: reorder from bucketed order and apply weighted sum
fn gather_and_finalize(
    y_partial: &[bf16],
    topk_probs: &[bf16],
    perm_idx: &[usize],
    t: usize,
    k: usize,
    d_model: usize,
) -> Vec<bf16> {
    let mut y = vec![bf16::from_f32(0.0); t * d_model];
    for tok in 0..t {
        for d in 0..d_model {
            let mut acc = 0.0f32;
            for ki in 0..k {
                let orig_row = tok * k + ki;
                let bucket_row = perm_idx[orig_row];
                let prob = f32::from(topk_probs[orig_row]);
                let val = f32::from(y_partial[bucket_row * d_model + d]);
                acc += prob * val;
            }
            y[tok * d_model + d] = bf16::from_f32(acc);
        }
    }
    y
}

/// Unified CPU reference for MoE computation
///
/// Computes: y[t] = Σ_k prob[t,k] * (activation(x[t] @ W13[expert_k]) @ W2[expert_k] + bias)
///
/// Supports 4 gating modes:
/// - 0: GELU(up)
/// - 1: SiLU(up)
/// - 2: SwiGLU = SiLU(gate) * up
/// - 3: GEGLU = GELU(gate) * up
#[allow(clippy::too_many_arguments)]
fn cpu_moe_reference(
    x: &[bf16],           // [T, d_model]
    topk_ids: &[i32],     // [T * K]
    topk_probs: &[bf16],  // [T * K]
    w13_all: &[bf16],     // [E, 2*d_ff, d_model] transposed
    w2_all: &[bf16],      // [E, d_model, d_ff] transposed
    up_biases: &[bf16],   // [E, 2*d_ff]
    down_biases: &[bf16], // [E, d_model]
    t: usize,
    d_model: usize,
    d_ff: usize,
    k: usize,
    gating_code: u32,
    silu_alpha: f32,
    gate_clip_min: f32,
    gate_clip_max: f32,
    up_clip_min: f32,
    up_clip_max: f32,
) -> Vec<bf16> {
    let mut y = vec![bf16::from_f32(0.0); t * d_model];

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

    // Process each token
    for tok in 0..t {
        let x_offset = tok * d_model;

        // Hidden buffer for this token's K experts [K, d_ff]
        let mut hidden = vec![0.0f32; k * d_ff];

        // Pass A: x @ W13[expert] -> hidden[k] with activation
        for k_idx in 0..k {
            let expert_id = topk_ids[tok * k + k_idx];
            if expert_id < 0 {
                continue;
            }
            let expert = expert_id as usize;

            let w13_base = expert * 2 * d_ff * d_model;
            let bias_base = expert * 2 * d_ff;

            for h in 0..d_ff {
                // Up projection
                let mut acc_up = f32::from(up_biases[bias_base + h]);
                for d in 0..d_model {
                    let x_val = f32::from(x[x_offset + d]);
                    let w_val = f32::from(w13_all[w13_base + h * d_model + d]);
                    acc_up += x_val * w_val;
                }

                // Apply activation based on gating_code
                let activated = if gating_code <= 1 {
                    // GELU or SiLU on up (with clipping)
                    let up_val = acc_up.clamp(up_clip_min, up_clip_max);
                    if gating_code == 0 {
                        gelu_approx(up_val)
                    } else {
                        silu(up_val, silu_alpha)
                    }
                } else {
                    // SwiGLU or GEGLU - need gate projection too
                    let mut acc_gate = f32::from(up_biases[bias_base + d_ff + h]);
                    for d in 0..d_model {
                        let x_val = f32::from(x[x_offset + d]);
                        let w_val = f32::from(w13_all[w13_base + (d_ff + h) * d_model + d]);
                        acc_gate += x_val * w_val;
                    }

                    // Apply clipping
                    let up_val = acc_up.clamp(up_clip_min, up_clip_max);
                    let gate_val = acc_gate.clamp(gate_clip_min, gate_clip_max);

                    let gate_act = if gating_code == 2 {
                        silu(gate_val, silu_alpha)
                    } else {
                        gelu_approx(gate_val)
                    };
                    gate_act * up_val
                };

                hidden[k_idx * d_ff + h] = activated;
            }
        }

        // Pass B: hidden @ W2 -> y (with weighted sum)
        for d in 0..d_model {
            let mut final_acc = 0.0f32;

            for k_idx in 0..k {
                let expert_id = topk_ids[tok * k + k_idx];
                if expert_id < 0 {
                    continue;
                }
                let expert = expert_id as usize;
                let prob = f32::from(topk_probs[tok * k + k_idx]);

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

            y[tok * d_model + d] = bf16::from_f32(final_acc);
        }
    }

    y
}

#[test]
fn test_two_pass_decode_correctness() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xDEC0DE42);

    // End-to-end test: T=1 token, K=2 experts per token
    let t = 1;
    let k = 2;
    let sum_k = t * k;
    let d_model = 512;
    let d_ff = 2048;
    let e = 8;
    let gating_code = 2u32; // SwiGLU
    let silu_alpha = 1.0f32;

    eprintln!("[2-pass decode] T={}, K={}, sum_k={}, d_model={}, d_ff={}, E={}", t, k, sum_k, d_model, d_ff, e);

    // Generate input x [T, d_model]
    let x: Vec<bf16> = (0..t * d_model).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect();

    // Generate routing: topk_ids [T*K], topk_probs [T*K]
    let topk_ids: Vec<i32> = (0..t * k).map(|i| ((i / k) + (i % k) * 2) as i32 % e as i32).collect();
    let topk_probs: Vec<bf16> = (0..t)
        .flat_map(|_| {
            let raw: Vec<f32> = (0..k).map(|_| rng.random_range(0.1..1.0)).collect();
            let sum: f32 = raw.iter().sum();
            raw.iter().map(|p| bf16::from_f32(p / sum)).collect::<Vec<_>>()
        })
        .collect();

    // Scatter x based on topk_ids to get x_perm
    // Each token contributes K rows, one per selected expert
    let mut x_perm = vec![bf16::from_f32(0.0); sum_k * d_model];
    for tok in 0..t {
        for ki in 0..k {
            let row_idx = tok * k + ki;
            for d in 0..d_model {
                x_perm[row_idx * d_model + d] = x[tok * d_model + d];
            }
        }
    }

    // Build offsets and row_expert_map based on topk_ids
    // First count rows per expert
    let mut expert_counts = vec![0usize; e];
    for &eid in &topk_ids {
        expert_counts[eid as usize] += 1;
    }
    let mut offsets = vec![0u32; e + 1];
    for i in 0..e {
        offsets[i + 1] = offsets[i] + expert_counts[i] as u32;
    }

    // Build row_expert_map (maps permuted row to expert)
    let mut row_expert_map = vec![0u32; sum_k];
    let mut expert_row_idx = vec![0usize; e];
    for (row, &eid) in topk_ids.iter().enumerate() {
        row_expert_map[row] = eid as u32;
        expert_row_idx[eid as usize] += 1;
    }

    // Generate weights
    let w13: Vec<bf16> = (0..e * 2 * d_ff * d_model).map(|_| bf16::from_f32(rng.random_range(-0.05..0.05))).collect();
    let w2: Vec<bf16> = (0..e * d_ff * d_model).map(|_| bf16::from_f32(rng.random_range(-0.05..0.05))).collect();
    let up_biases: Vec<bf16> = (0..e * 2 * d_ff).map(|_| bf16::from_f32(rng.random_range(-0.01..0.01))).collect();
    let down_biases: Vec<bf16> = (0..e * d_model).map(|_| bf16::from_f32(rng.random_range(-0.01..0.01))).collect();

    // CPU reference - end-to-end
    let y_expected = cpu_moe_reference(
        &x,
        &topk_ids,
        &topk_probs,
        &w13,
        &w2,
        &up_biases,
        &down_biases,
        t,
        d_model,
        d_ff,
        k,
        gating_code,
        silu_alpha,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
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
    let y_partial_buf = alloc_buffer::<bf16>(&ctx, sum_k * d_model);

    // Tile infrastructure
    let h_blocks_decode = (d_ff + 3) / 4;
    let max_total_tiles = sum_k * h_blocks_decode;

    let tile_counts_buf = alloc_buffer::<u32>(&ctx, e);
    let tile_offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let total_tiles_buf = alloc_buffer::<u32>(&ctx, 8);
    let tile_map_buf = alloc_buffer::<u32>(&ctx, max_total_tiles * 3);
    let dispatch_args_buf = alloc_buffer::<u32>(&ctx, 3);

    // Execute 2-pass decode kernel
    let experts_kernel = MoeExpertsTwoPassDecodeKernels::new(&ctx).expect("MoeExpertsTwoPassDecodeKernel::new");
    let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");

    const K_TILE: usize = 64;
    let num_tiles_k = ((d_ff + K_TILE - 1) / K_TILE) as u32;

    experts_kernel.encode(
        &cb,
        &MoeExpertsTwoPassArguments {
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
            num_tiles_k,
            gating_code,
            gate_clip_min: f32::NEG_INFINITY,
            gate_clip_max: f32::INFINITY,
            up_clip_min: f32::NEG_INFINITY,
            up_clip_max: f32::INFINITY,
            silu_alpha,
            data_type: DataType::BF16,
        },
    );

    cb.commit();
    cb.wait_until_completed();

    // Read GPU partial output and do CPU finalize (weighted sum)
    let y_partial_gpu =
        unsafe { std::slice::from_raw_parts(y_partial_buf.contents().as_ptr() as *const bf16, sum_k * d_model) };

    // Finalize: y[t] = Σ_k prob[t,k] * y_partial[t*k + k_idx]
    let mut y_gpu = vec![bf16::from_f32(0.0); t * d_model];
    for tok in 0..t {
        for d in 0..d_model {
            let mut acc = 0.0f32;
            for ki in 0..k {
                let row_idx = tok * k + ki;
                let prob = f32::from(topk_probs[row_idx]);
                let val = f32::from(y_partial_gpu[row_idx * d_model + d]);
                acc += prob * val;
            }
            y_gpu[tok * d_model + d] = bf16::from_f32(acc);
        }
    }

    // Compute error metrics
    let mut max_abs_error = 0.0f32;
    let mut max_idx = 0;

    for (i, (&gpu_val, &cpu_val)) in y_gpu.iter().zip(y_expected.iter()).enumerate() {
        let abs_error = (f32::from(gpu_val) - f32::from(cpu_val)).abs();
        if abs_error > max_abs_error {
            max_abs_error = abs_error;
            max_idx = i;
        }
    }

    eprintln!(
        "[2-pass decode] Max error: {:.6} at idx {} (GPU={:.6}, CPU={:.6})",
        max_abs_error,
        max_idx,
        f32::from(y_gpu[max_idx]),
        f32::from(y_expected[max_idx])
    );

    let tolerance = 0.02;
    assert_bf16_close(&y_gpu, &y_expected, tolerance, "2-pass decode output");

    eprintln!("[2-pass decode] ✓ PASSED (tolerance={:.4})", tolerance);
}

#[test]
fn test_two_pass_decode_multi_token() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xDEC0DE43);

    let t = 4;
    let k = 2;
    let sum_k = t * k;
    let d_model = 512;
    let d_ff = 2048;
    let e = 8;
    let gating_code = 2u32;
    let silu_alpha = 1.0f32;

    eprintln!("[2-pass decode multi-token] T={}, K={}, d_model={}, d_ff={}, E={}", t, k, d_model, d_ff, e);

    let data = MoeTestData::generate(&mut rng, t, k, d_model, d_ff, e);
    let scatter = scatter_by_expert(&data.x, &data.topk_ids, t, k, d_model, e);

    let y_expected = cpu_moe_reference(
        &data.x,
        &data.topk_ids,
        &data.topk_probs,
        &data.w13,
        &data.w2,
        &data.up_biases,
        &data.down_biases,
        t,
        d_model,
        d_ff,
        k,
        gating_code,
        silu_alpha,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
    );

    // GPU buffers
    let x_perm_buf = alloc_buffer_with_data(&ctx, &scatter.x_perm);
    let offsets_buf = alloc_buffer_with_data(&ctx, &scatter.offsets);
    let row_expert_map_buf = alloc_buffer_with_data(&ctx, &scatter.row_expert_map);
    let w13_buf = alloc_buffer_with_data(&ctx, &data.w13);
    let w2_buf = alloc_buffer_with_data(&ctx, &data.w2);
    let up_biases_buf = alloc_buffer_with_data(&ctx, &data.up_biases);
    let down_biases_buf = alloc_buffer_with_data(&ctx, &data.down_biases);
    let hidden_buf = alloc_buffer::<f32>(&ctx, sum_k * d_ff);
    let y_partial_buf = alloc_buffer::<bf16>(&ctx, sum_k * d_model);

    // Tile infrastructure
    let max_total_tiles = sum_k * ((d_ff + 3) / 4);
    let tile_counts_buf = alloc_buffer::<u32>(&ctx, e);
    let tile_offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let total_tiles_buf = alloc_buffer::<u32>(&ctx, 8);
    let tile_map_buf = alloc_buffer::<u32>(&ctx, max_total_tiles * 3);
    let dispatch_args_buf = alloc_buffer::<u32>(&ctx, 3);

    let experts_kernel = MoeExpertsTwoPassDecodeKernels::new(&ctx).expect("kernel");
    let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
    experts_kernel.encode(
        &cb,
        &MoeExpertsTwoPassArguments {
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
            num_tiles_k: ((d_ff + 63) / 64) as u32,
            gating_code,
            gate_clip_min: f32::NEG_INFINITY,
            gate_clip_max: f32::INFINITY,
            up_clip_min: f32::NEG_INFINITY,
            up_clip_max: f32::INFINITY,
            silu_alpha,
            data_type: DataType::BF16,
        },
    );
    cb.commit();
    cb.wait_until_completed();

    let y_partial_gpu =
        unsafe { std::slice::from_raw_parts(y_partial_buf.contents().as_ptr() as *const bf16, sum_k * d_model) };
    let y_gpu = gather_and_finalize(y_partial_gpu, &data.topk_probs, &scatter.perm_idx, t, k, d_model);

    assert_bf16_close(&y_gpu, &y_expected, 0.02, "2-pass decode multi-token");
    eprintln!("[2-pass decode multi-token] ✓ PASSED");
}

#[test]
fn test_two_pass_prefill_correctness() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(0xFE1111);

    let t = 8;
    let k = 2;
    let sum_k = t * k;
    let d_model = 512;
    let d_ff = 2048;
    let e = 8;
    let gating_code = 2u32;
    let silu_alpha = 1.0f32;

    eprintln!("[2-pass prefill] T={}, K={}, d_model={}, d_ff={}, E={}", t, k, d_model, d_ff, e);

    let data = MoeTestData::generate(&mut rng, t, k, d_model, d_ff, e);
    let scatter = scatter_by_expert(&data.x, &data.topk_ids, t, k, d_model, e);

    let y_expected = cpu_moe_reference(
        &data.x,
        &data.topk_ids,
        &data.topk_probs,
        &data.w13,
        &data.w2,
        &data.up_biases,
        &data.down_biases,
        t,
        d_model,
        d_ff,
        k,
        gating_code,
        silu_alpha,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
    );

    // GPU buffers
    let x_perm_buf = alloc_buffer_with_data(&ctx, &scatter.x_perm);
    let offsets_buf = alloc_buffer_with_data(&ctx, &scatter.offsets);
    let row_expert_map_buf = alloc_buffer_with_data(&ctx, &scatter.row_expert_map);
    let w13_buf = alloc_buffer_with_data(&ctx, &data.w13);
    let w2_buf = alloc_buffer_with_data(&ctx, &data.w2);
    let up_biases_buf = alloc_buffer_with_data(&ctx, &data.up_biases);
    let down_biases_buf = alloc_buffer_with_data(&ctx, &data.down_biases);
    let hidden_buf = alloc_buffer::<f32>(&ctx, sum_k * d_ff);
    let y_partial_buf = alloc_buffer::<bf16>(&ctx, sum_k * d_model);

    // Tile infrastructure
    let max_total_tiles = sum_k * ((d_ff + 3) / 4);
    let tile_counts_buf = alloc_buffer::<u32>(&ctx, e);
    let tile_offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let total_tiles_buf = alloc_buffer::<u32>(&ctx, 8);
    let tile_map_buf = alloc_buffer::<u32>(&ctx, max_total_tiles * 3);
    let dispatch_args_buf = alloc_buffer::<u32>(&ctx, 3);

    let experts_kernel = MoeExpertsTwoPassPrefillKernel::new(&ctx).expect("kernel");
    let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
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
                num_tiles_k: ((d_ff + 63) / 64) as u32,
                gating_code,
                gate_clip_min: f32::NEG_INFINITY,
                gate_clip_max: f32::INFINITY,
                up_clip_min: f32::NEG_INFINITY,
                up_clip_max: f32::INFINITY,
                silu_alpha,
                data_type: DataType::BF16,
            },
        )
        .expect("encode");
    cb.commit();
    cb.wait_until_completed();

    let y_partial_gpu =
        unsafe { std::slice::from_raw_parts(y_partial_buf.contents().as_ptr() as *const bf16, sum_k * d_model) };
    let y_gpu = gather_and_finalize(y_partial_gpu, &data.topk_probs, &scatter.perm_idx, t, k, d_model);

    assert_bf16_close(&y_gpu, &y_expected, 0.02, "2-pass prefill");
    eprintln!("[2-pass prefill] ✓ PASSED");
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

    eprintln!("[fused single-token] d_model={}, d_ff={}, E={}, K={}", d_model, d_ff, e, k);

    // Generate test data
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

    // CPU reference (gating_code=2 for SwiGLU)
    let gating_code = 2u32;
    let silu_alpha = 1.0f32;
    let y_expected = cpu_moe_reference(
        &x,
        &topk_ids,
        &topk_probs,
        &w13_all,
        &w2_all,
        &up_biases,
        &down_biases,
        1, // T=1 for single token
        d_model,
        d_ff,
        k,
        gating_code,
        silu_alpha,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
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
    let fused_kernel = MoeExpertsSingleDecodeKernels::new(&ctx).expect("MoeExpertsSingleDecodeKernel::new");
    let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");

    fused_kernel.encode(
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
            silu_alpha: 1.0, // Standard SiLU for testing
            gate_clip_min: f32::NEG_INFINITY,
            gate_clip_max: f32::INFINITY,
            up_clip_min: f32::NEG_INFINITY,
            up_clip_max: f32::INFINITY,
            data_type: DataType::BF16,
        },
    );

    cb.commit();
    cb.wait_until_completed();

    // Read GPU output
    let y_gpu = unsafe { std::slice::from_raw_parts(y_buf.contents().as_ptr() as *const bf16, d_model) };

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
        max_abs_error,
        max_idx,
        f32::from(y_gpu[max_idx]),
        f32::from(y_expected[max_idx]),
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

    eprintln!("[fused single-token K=4] d_model={}, d_ff={}, E={}, K={}", d_model, d_ff, e, k);

    let x: Vec<bf16> = (0..d_model).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect();

    let topk_ids: Vec<i32> = (0..k).map(|i| ((i * 2) % e) as i32).collect();

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

    let gating_code = 2u32;
    let silu_alpha = 1.0f32;
    let y_expected = cpu_moe_reference(
        &x,
        &topk_ids,
        &topk_probs,
        &w13_all,
        &w2_all,
        &up_biases,
        &down_biases,
        1, // T=1 for single token
        d_model,
        d_ff,
        k,
        gating_code,
        silu_alpha,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
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

    let fused_kernel = MoeExpertsSingleDecodeKernels::new(&ctx).expect("fused kernel");
    let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
    fused_kernel.encode(
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
            silu_alpha,
            gate_clip_min: f32::NEG_INFINITY,
            gate_clip_max: f32::INFINITY,
            up_clip_min: f32::NEG_INFINITY,
            up_clip_max: f32::INFINITY,
            data_type: DataType::BF16,
        },
    );
    cb.commit();
    cb.wait_until_completed();

    let y_gpu = unsafe { std::slice::from_raw_parts(y_buf.contents().as_ptr() as *const bf16, d_model) };

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
        max_abs_error,
        max_idx,
        f32::from(y_gpu[max_idx]),
        f32::from(y_expected[max_idx]),
        mean_abs_error
    );

    let tolerance = 0.02;
    assert_bf16_close(y_gpu, &y_expected, tolerance, "fused single-token K=4 output");

    eprintln!("[fused single-token K=4] PASSED (tolerance={:.4})", tolerance);
}
