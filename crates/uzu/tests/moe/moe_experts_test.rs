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
    },
};

#[path = "moe_test_utils.rs"]
mod test_utils;
use test_utils::{
    alloc_buffer, alloc_buffer_with_data, assert_bf16_close, cpu_tile_counts,
    cpu_tile_scan, create_ctx,
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
        let mut hidden = vec![bf16::from_f32(0.0); expert_rows * 2 * d_ff];
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
                hidden[row * 2 * d_ff + ff] = bf16::from_f32(acc);
            }
        }

        // Apply activation based on gating_code
        for row in 0..expert_rows {
            for ff in 0..d_ff {
                let up = f32::from(hidden[row * 2 * d_ff + ff]);

                let result = if gating_code <= 1 {
                    // GELU(up) or SiLU(up) - no gating
                    if gating_code == 0 {
                        gelu_approx(up)
                    } else {
                        silu(up, silu_alpha)
                    }
                } else {
                    // SwiGLU or GEGLU - gate * up
                    let gate = f32::from(hidden[row * 2 * d_ff + d_ff + ff]);
                    let activated_gate = if gating_code == 2 {
                        silu(gate, silu_alpha)
                    } else {
                        gelu_approx(gate)
                    };
                    activated_gate * up
                };

                hidden[row * 2 * d_ff + ff] = bf16::from_f32(result);
                hidden_full[(row_start + row) * d_ff + ff] = result;
            }
        }

        // FC2: [expert_rows, d_ff] @ [d_ff, d_model] -> [expert_rows, d_model]
        for row in 0..expert_rows {
            for dm in 0..d_model {
                let mut acc = f32::from(down_biases[expert * d_model + dm]);
                for ff in 0..d_ff {
                    let h_val = f32::from(hidden[row * 2 * d_ff + ff]);
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
