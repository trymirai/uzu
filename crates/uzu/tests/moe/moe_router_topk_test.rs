#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::bf16;
use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::{
    DataType,
    backends::{
        common::kernel::MoeRouterTopKKernel,
        metal::{MetalContext, kernel::dsl::MoeRouterTopKMetalKernel},
    },
};

use super::test_utils::{alloc_buffer, alloc_buffer_with_data, create_ctx};

/// CPU reference for router logits (bf16 precision, f32 accumulation).
pub fn cpu_router_logits_bf16(
    input: &[bf16],
    weight: &[bf16],
    bias: &[bf16],
    t: usize,
    e: usize,
    d_model: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), t * d_model);
    assert_eq!(weight.len(), e * d_model);
    assert_eq!(bias.len(), e);
    assert_eq!(d_model % 4, 0, "d_model must be multiple of 4");

    let mut out = vec![0.0f32; t * e];
    for token in 0..t {
        let x_row = &input[token * d_model..(token + 1) * d_model];
        for expert in 0..e {
            let w_row = &weight[expert * d_model..(expert + 1) * d_model];

            // Simulate GPU vec4 processing: accumulate in 4-element chunks
            let mut accum = [0.0f32; 4];
            for chunk in (0..d_model).step_by(4) {
                // Load vec4
                let xv = [
                    f32::from(x_row[chunk]),
                    f32::from(x_row[chunk + 1]),
                    f32::from(x_row[chunk + 2]),
                    f32::from(x_row[chunk + 3]),
                ];
                let wv = [
                    f32::from(w_row[chunk]),
                    f32::from(w_row[chunk + 1]),
                    f32::from(w_row[chunk + 2]),
                    f32::from(w_row[chunk + 3]),
                ];
                // FMA: accum = wv * xv + accum
                for i in 0..4 {
                    accum[i] = wv[i].mul_add(xv[i], accum[i]);
                }
            }

            // Sum the 4-vector: (a.x + a.y) + (a.z + a.w) - matches Metal shader line 60
            let sum = (accum[0] + accum[1]) + (accum[2] + accum[3]);
            out[token * e + expert] = sum + f32::from(bias[expert]);
        }
    }
    out
}

/// CPU reference for Top-K selection over float32 logits.
pub fn cpu_topk_select_f32(
    logits: &[f32],
    t: usize,
    e: usize,
    k: usize,
    renorm: bool,
) -> (Vec<i32>, Vec<f32>) {
    assert_eq!(logits.len(), t * e);
    assert!(k >= 1 && e >= k);
    let mut ids = vec![0i32; t * k];
    let mut probs = vec![0f32; t * k];
    for token in 0..t {
        let row = &logits[token * e..(token + 1) * e];
        let mut best_vals = vec![f32::NEG_INFINITY; k];
        let mut best_ids = vec![-1i32; k];
        for expert in 0..e {
            let v = row[expert];
            let mut insert_pos = None;
            for j in (0..k).rev() {
                if v > best_vals[j] || (v == best_vals[j] && (expert as i32) < best_ids[j]) {
                    insert_pos = Some(j);
                }
            }
            if let Some(pos) = insert_pos {
                for s in (pos + 1..k).rev() {
                    best_vals[s] = best_vals[s - 1];
                    best_ids[s] = best_ids[s - 1];
                }
                best_vals[pos] = v;
                best_ids[pos] = expert as i32;
            }
        }
        let base = token * k;
        for kk in 0..k {
            ids[base + kk] = best_ids[kk];
        }
        if renorm {
            let max_v = best_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exps = vec![0.0f32; k];
            let mut sum = 0.0f32;
            for kk in 0..k {
                exps[kk] = (best_vals[kk] - max_v).exp();
                sum += exps[kk];
            }
            if sum > 0.0 {
                for kk in 0..k {
                    probs[base + kk] = exps[kk] / sum;
                }
            } else {
                let uniform = 1.0f32 / k as f32;
                for kk in 0..k {
                    probs[base + kk] = uniform;
                }
            }
        } else {
            for kk in 0..k {
                probs[base + kk] = f32::from(bf16::from_f32(best_vals[kk]));
            }
        }
    }
    (ids, probs)
}

fn run_router_topk_once(
    ctx: &MetalContext,
    kernel: &MoeRouterTopKMetalKernel,
    t: usize,
    d_model: usize,
    e: usize,
    k: usize,
    renorm: bool,
) {
    let mut rng = StdRng::seed_from_u64(1234);
    let input_f32: Vec<f32> = (0..t * d_model).map(|_| rng.random_range(-1.0..1.0)).collect();
    let weight_f32: Vec<f32> = (0..e * d_model).map(|_| rng.random_range(-1.0..1.0)).collect();
    let bias_f32: Vec<f32> = (0..e).map(|_| rng.random_range(-0.5..0.5)).collect();

    // Convert to bf16
    let input: Vec<bf16> = input_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let weight: Vec<bf16> = weight_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let bias: Vec<bf16> = bias_f32.iter().map(|&x| bf16::from_f32(x)).collect();

    // Compute CPU reference using bf16 inputs (with f32 accumulation)
    let logits_ref = cpu_router_logits_bf16(&input, &weight, &bias, t, e, d_model);
    let (ids_ref, probs_ref) = cpu_topk_select_f32(&logits_ref, t, e, k, renorm);

    let input_buf = alloc_buffer_with_data(ctx, &input);
    let weight_buf = alloc_buffer_with_data(ctx, &weight);
    let bias_buf = alloc_buffer_with_data(ctx, &bias);
    let ids_buf = alloc_buffer::<i32>(ctx, t * k);
    // For BFloat16 kernel, probs buffer must be bf16, not f32
    let probs_buf = alloc_buffer::<bf16>(ctx, t * k);

    let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
    let encoder = cb.new_compute_command_encoder().expect("Failed to create command encoder");
    kernel.encode(
        &input_buf,
        &weight_buf,
        &bias_buf,
        &ids_buf,
        &probs_buf,
        t as u32,
        d_model as u32,
        e as u32,
        k as u32,
        renorm,
        &encoder,
    );
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let ids_ptr = ids_buf.contents().as_ptr() as *const i32;
    let probs_ptr = probs_buf.contents().as_ptr() as *const bf16;
    let ids_gpu = unsafe { std::slice::from_raw_parts(ids_ptr, t * k) }.to_vec();
    let probs_bf16_gpu = unsafe { std::slice::from_raw_parts(probs_ptr, t * k) }.to_vec();
    // Convert bf16 to f32 for comparison
    let probs_gpu: Vec<f32> = probs_bf16_gpu.iter().map(|&h| f32::from(h)).collect();

    assert_eq!(
        ids_gpu, ids_ref,
        "Top-k ids mismatch for T={}, d_model={}, E={}, K={}, renorm={}",
        t, d_model, e, k, renorm
    );

    for i in 0..(t * k) {
        let diff = (probs_gpu[i] - probs_ref[i]).abs();

        // Tolerance accounts for numerical differences in:
        // 1. Router matmul: GPU uses vec4 FMA + SIMD reductions, CPU uses scalar loops
        //    With bf16 inputs and f32 accumulation, minor differences expected
        // 2. bf16 quantization on probs output buffer
        // 3. TopK selection (minor)
        // 4. Softmax computation when renorm=true (exp/div operations)
        let atol = if renorm {
            2e-2 // Normalized probabilities with bf16: softmax + bf16 quantization
        } else {
            5e-2 // Raw logits with bf16: vec4 accumulation + bf16 output quantization
        };
        assert!(
            diff <= atol,
            "Top-k prob mismatch at {}: gpu={} ref={} (diff={}, atol={}) with T={}, d_model={}, E={}, K={}, renorm={}",
            i,
            probs_gpu[i],
            probs_ref[i],
            diff,
            atol,
            t,
            d_model,
            e,
            k,
            renorm
        );
    }
}

#[test]
fn test_router_topk_fused_matches_reference() {
    let ctx = create_ctx();
    let kernel = MoeRouterTopKMetalKernel::new(&ctx, DataType::BF16).expect("kernel");

    let configs =
        [(1usize, 64usize, 32usize, 4usize), (2, 128, 64, 8), (4, 256, 128, 16), (8, 256, 256, 32), (1, 512, 512, 64)];
    let renorm_options = [false, true];

    for &(t, d_model, e, k) in &configs {
        for &renorm in &renorm_options {
            run_router_topk_once(&ctx, &kernel, t, d_model, e, k, renorm);
        }
    }
}
