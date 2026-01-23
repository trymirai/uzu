#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::bf16;
use metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    MTLContext, ProtocolObject, Retained,
    kernel::{
        KernelDataType, MoeCountsOffsetsFusedArguments,
        MoeCountsOffsetsFusedKernel,
        moe::{MoeRouterTopKArguments, MoeRouterTopKKernel},
    },
};

use super::test_utils::{alloc_buffer, alloc_buffer_with_data, create_ctx};

fn cpu_bucket_counts(
    topk_ids: &[i32],
    t: usize,
    k: usize,
    e: usize,
) -> Vec<u32> {
    let mut counts = vec![0u32; e];
    for ti in 0..t {
        for kk in 0..k {
            let id = topk_ids[ti * k + kk];
            if id >= 0 {
                let ue = id as usize;
                if ue < e {
                    counts[ue] += 1;
                }
            }
        }
    }
    counts
}

fn cpu_offsets_from_counts(counts: &[u32]) -> (Vec<u32>, u32) {
    let mut offsets = vec![0u32; counts.len() + 1];
    let mut sum = 0u32;
    for (i, &c) in counts.iter().enumerate() {
        offsets[i] = sum;
        sum += c;
    }
    offsets[counts.len()] = sum;
    (offsets, sum)
}

fn gen_topk_ids_from_logits(
    ctx: &MTLContext,
    t: usize,
    e: usize,
    k: usize,
) -> (Vec<i32>, Retained<ProtocolObject<dyn MTLBuffer>>) {
    let mut rng = StdRng::seed_from_u64(1234);

    // Generate random input and router weights
    let d_model = 64; // arbitrary model dimension for test
    let input_f32: Vec<f32> =
        (0..t * d_model).map(|_| rng.random_range(-1.0..1.0)).collect();
    let weight_f32: Vec<f32> =
        (0..e * d_model).map(|_| rng.random_range(-1.0..1.0)).collect();
    let bias_f32: Vec<f32> =
        (0..e).map(|_| rng.random_range(-0.5..0.5)).collect();

    // Convert to bf16
    let input: Vec<bf16> =
        input_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let weight: Vec<bf16> =
        weight_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let bias: Vec<bf16> = bias_f32.iter().map(|&x| bf16::from_f32(x)).collect();

    let input_buf = alloc_buffer_with_data(ctx, &input);
    let weight_buf = alloc_buffer_with_data(ctx, &weight);
    let bias_buf = alloc_buffer_with_data(ctx, &bias);
    let topk_ids_buf = alloc_buffer::<i32>(ctx, t * k);
    let topk_probs_buf = alloc_buffer::<bf16>(ctx, t * k);

    // Use fused router+topk kernel
    let router_topk =
        MoeRouterTopKKernel::new(ctx).expect("router_topk kernel");
    let cb = ctx
        .command_queue
        .command_buffer()
        .expect("Failed to create command buffer");
    let args = MoeRouterTopKArguments {
        input_buffer: &input_buf,
        weight_buffer: &weight_buf,
        bias_buffer: &bias_buf,
        topk_ids_buffer: &topk_ids_buf,
        topk_probs_buffer: &topk_probs_buf,
        t,
        d_model,
        e,
        k,
        renorm: true,
    };
    router_topk
        .encode(&cb, KernelDataType::BFloat16, args)
        .expect("encode router_topk");
    cb.commit();
    cb.wait_until_completed();

    let ids_ptr = topk_ids_buf.contents().as_ptr() as *const i32;
    let ids = unsafe { std::slice::from_raw_parts(ids_ptr, t * k) };
    (ids.to_vec(), topk_ids_buf)
}

#[test]
fn test_counts_offsets_fused_parity_random() {
    let ctx = create_ctx();
    let shapes = vec![(1usize, 4usize), (7, 16), (64, 64), (1024, 128)];
    let ks = vec![1usize, 2usize, 4usize];

    for &(t, e) in &shapes {
        for &k in &ks {
            if k > e {
                continue;
            }
            let (topk_ids, topk_ids_buf) =
                gen_topk_ids_from_logits(&ctx, t, e, k);

            // CPU reference
            let counts_cpu = cpu_bucket_counts(&topk_ids, t, k, e);
            let (offsets_cpu, sum_cpu) = cpu_offsets_from_counts(&counts_cpu);

            // GPU fused kernel
            let offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
            let sum_k_buf = alloc_buffer::<u32>(&ctx, 1);
            let num_tiles = ((e + 511) / 512).max(1);
            let partials_buf = alloc_buffer::<u32>(&ctx, num_tiles * 512);

            let kernel =
                MoeCountsOffsetsFusedKernel::new(&ctx).expect("fused kernel");
            let cb = ctx
                .command_queue
                .command_buffer()
                .expect("Failed to create command buffer");
            let args = MoeCountsOffsetsFusedArguments {
                topk_ids_buffer: &topk_ids_buf,
                offsets_buffer: &offsets_buf,
                sum_k_buffer: &sum_k_buf,
                partials_buffer: &partials_buf,
                t,
                e,
                k,
            };
            kernel.encode(&cb, args).expect("encode fused");
            cb.commit();
            cb.wait_until_completed();

            // Verify offsets
            let offsets_ptr = offsets_buf.contents().as_ptr() as *const u32;
            let offsets_gpu =
                unsafe { std::slice::from_raw_parts(offsets_ptr, e + 1) };
            assert_eq!(
                offsets_gpu,
                &offsets_cpu[..],
                "offsets mismatch T={} E={} K={}",
                t,
                e,
                k
            );

            // Verify sum
            let sum_ptr = sum_k_buf.contents().as_ptr() as *const u32;
            let sum_gpu = unsafe { *sum_ptr };
            assert_eq!(
                sum_gpu, sum_cpu,
                "sum mismatch T={} E={} K={}",
                t, e, k
            );
        }
    }
}

#[test]
fn test_counts_offsets_fused_edge_cases() {
    let ctx = create_ctx();

    // Case 1: All tokens to one expert
    let (t, e, k) = (16usize, 8usize, 2usize);
    let mut topk_ids = vec![-1i32; t * k];
    for ti in 0..t {
        for kk in 0..k {
            topk_ids[ti * k + kk] = 3;
        }
    }
    let topk_ids_buf = alloc_buffer_with_data(&ctx, &topk_ids);

    let offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let sum_k_buf = alloc_buffer::<u32>(&ctx, 1);
    let num_tiles = ((e + 511) / 512).max(1);
    let partials_buf = alloc_buffer::<u32>(&ctx, num_tiles * 512);

    let kernel = MoeCountsOffsetsFusedKernel::new(&ctx).expect("fused kernel");
    let cb = ctx
        .command_queue
        .command_buffer()
        .expect("Failed to create command buffer");
    let args = MoeCountsOffsetsFusedArguments {
        topk_ids_buffer: &topk_ids_buf,
        offsets_buffer: &offsets_buf,
        sum_k_buffer: &sum_k_buf,
        partials_buffer: &partials_buf,
        t,
        e,
        k,
    };
    kernel.encode(&cb, args).expect("encode fused");
    cb.commit();
    cb.wait_until_completed();

    let offsets_gpu = unsafe {
        std::slice::from_raw_parts(
            offsets_buf.contents().as_ptr() as *const u32,
            e + 1,
        )
    };

    let mut expected_offsets = vec![0u32; e + 1];
    expected_offsets[4..].fill((t * k) as u32);
    assert_eq!(offsets_gpu, &expected_offsets[..]);

    // Case 2: T=0
    let (t, e, k) = (0usize, 8usize, 2usize);
    let topk_ids: Vec<i32> = vec![];
    let topk_ids_buf = alloc_buffer_with_data(&ctx, &topk_ids);
    let offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let sum_k_buf = alloc_buffer::<u32>(&ctx, 1);
    let num_tiles = ((e + 511) / 512).max(1);
    let partials_buf = alloc_buffer::<u32>(&ctx, num_tiles * 512);

    let kernel = MoeCountsOffsetsFusedKernel::new(&ctx).expect("fused kernel");
    let cb = ctx
        .command_queue
        .command_buffer()
        .expect("Failed to create command buffer");
    let args = MoeCountsOffsetsFusedArguments {
        topk_ids_buffer: &topk_ids_buf,
        offsets_buffer: &offsets_buf,
        sum_k_buffer: &sum_k_buf,
        partials_buffer: &partials_buf,
        t,
        e,
        k,
    };
    kernel.encode(&cb, args).expect("encode fused");
    cb.commit();
    cb.wait_until_completed();

    let offsets_gpu = unsafe {
        std::slice::from_raw_parts(
            offsets_buf.contents().as_ptr() as *const u32,
            e + 1,
        )
    };
    assert!(offsets_gpu.iter().all(|&v| v == 0));
}
