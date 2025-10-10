#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::bf16;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::moe::{MoeRouterTopKArguments, MoeRouterTopKKernel},
};

use super::test_utils::{
    alloc_buffer, alloc_buffer_with_data, cpu_router_logits_bf16,
    cpu_topk_select_f32, create_ctx,
};

fn run_router_topk_once(
    ctx: &MTLContext,
    kernel: &MoeRouterTopKKernel,
    t: usize,
    d_model: usize,
    e: usize,
    k: usize,
    renorm: bool,
) {
    let mut rng = StdRng::seed_from_u64(1234);
    let input_f32: Vec<f32> =
        (0..t * d_model).map(|_| rng.random_range(-1.0..1.0)).collect();
    let weight_f32: Vec<f32> =
        (0..e * d_model).map(|_| rng.random_range(-1.0..1.0)).collect();
    let bias_f32: Vec<f32> = (0..e).map(|_| rng.random_range(-0.5..0.5)).collect();

    // Convert to bf16
    let input: Vec<bf16> = input_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let weight: Vec<bf16> = weight_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let bias: Vec<bf16> = bias_f32.iter().map(|&x| bf16::from_f32(x)).collect();

    // Compute CPU reference using bf16 inputs (with f32 accumulation)
    let logits_ref =
        cpu_router_logits_bf16(&input, &weight, &bias, t, e, d_model);
    let (ids_ref, probs_ref) =
        cpu_topk_select_f32(&logits_ref, t, e, k, renorm);

    // Debug: print first few logits if ENV var is set
    if std::env::var_os("DEBUG_ROUTER_TOPK").is_some() && t == 1 && e == 32 {
        eprintln!("CPU logits (ALL): {:?}", &logits_ref);
        eprintln!("CPU top-K ids: {:?}", &ids_ref[..k]);
        eprintln!("CPU top-K probs: {:?}", &probs_ref[..k]);
        // Show which experts were selected
        for kk in 0..k {
            let expert_id = ids_ref[kk] as usize;
            if expert_id < e {
                eprintln!("  Expert {}: logit={}", expert_id, logits_ref[expert_id]);
            }
        }
    }

    let input_buf = alloc_buffer_with_data(ctx, &input);
    let weight_buf = alloc_buffer_with_data(ctx, &weight);
    let bias_buf = alloc_buffer_with_data(ctx, &bias);
    let ids_buf = alloc_buffer::<i32>(ctx, t * k);
    // For BFloat16 kernel, probs buffer must be bf16, not f32
    let probs_buf = alloc_buffer::<bf16>(ctx, t * k);

    let cb = ctx.command_queue.new_command_buffer();
    let args = MoeRouterTopKArguments {
        input_buffer: &input_buf,
        weight_buffer: &weight_buf,
        bias_buffer: &bias_buf,
        topk_ids_buffer: &ids_buf,
        topk_probs_buffer: &probs_buf,
        t,
        d_model,
        e,
        k,
        renorm,
    };
    kernel
        .encode(&cb, KernelDataType::BFloat16, args)
        .expect("encode fused router+topk");
    cb.commit();
    cb.wait_until_completed();

    let ids_ptr = ids_buf.contents() as *const i32;
    let probs_ptr = probs_buf.contents() as *const bf16;
    let ids_gpu =
        unsafe { std::slice::from_raw_parts(ids_ptr, t * k) }.to_vec();
    let probs_bf16_gpu =
        unsafe { std::slice::from_raw_parts(probs_ptr, t * k) }.to_vec();
    // Convert bf16 to f32 for comparison
    let probs_gpu: Vec<f32> = probs_bf16_gpu.iter().map(|&h| f32::from(h)).collect();

    // Debug: print GPU results if ENV var is set
    if std::env::var_os("DEBUG_ROUTER_TOPK").is_some() && t == 1 && e == 32 {
        eprintln!("GPU top-K ids: {:?}", &ids_gpu[..k]);
        eprintln!("GPU top-K probs: {:?}", &probs_gpu[..k]);
    }

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
            2e-2  // Normalized probabilities with bf16: softmax + bf16 quantization
        } else {
            5e-2  // Raw logits with bf16: vec4 accumulation + bf16 output quantization
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
    let kernel = MoeRouterTopKKernel::new(&ctx).expect("kernel");

    let configs = [
        (1usize, 64usize, 32usize, 4usize),
        (2, 128, 64, 8),
        (4, 256, 128, 16),
        (8, 256, 256, 32),
        (1, 512, 512, 64),
    ];
    let renorm_options = [false, true];

    for &(t, d_model, e, k) in &configs {
        for &renorm in &renorm_options {
            run_router_topk_once(&ctx, &kernel, t, d_model, e, k, renorm);
        }
    }
}
