#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::f16;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    MTLContext,
    kernel::{
        KernelDataType, MoeCountsOffsetsFusedArguments,
        MoeCountsOffsetsFusedKernel, MoeTopKArguments, MoeTopKKernel,
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
) -> (Vec<i32>, metal::Buffer) {
    let mut rng = StdRng::seed_from_u64(1234);
    let logits: Vec<f32> =
        (0..t * e).map(|_| rng.random_range(-4.0..4.0)).collect();

    let logits_buf = alloc_buffer_with_data(ctx, &logits);

    let topk_ids_buf = alloc_buffer::<i32>(ctx, t * k);
    let topk_probs_buf = alloc_buffer::<f16>(ctx, t * k);

    let topk = MoeTopKKernel::new(ctx).expect("topk kernel");
    let cb = ctx.command_queue.new_command_buffer();
    let args = MoeTopKArguments {
        logits_buffer: &logits_buf,
        topk_ids_buffer: &topk_ids_buf,
        topk_probs_buffer: &topk_probs_buf,
        t,
        e,
        k,
        renorm: true,
    };
    topk.encode(&cb, KernelDataType::Float32, args).expect("encode topk");
    cb.commit();
    cb.wait_until_completed();

    let ids_ptr = topk_ids_buf.contents() as *const i32;
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
            let counts_buf = alloc_buffer::<u32>(&ctx, e);
            let offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
            let sum_k_buf = alloc_buffer::<u32>(&ctx, 1);
            let num_tiles = ((e + 511) / 512).max(1);
            let partials_buf = alloc_buffer::<u32>(&ctx, num_tiles * 512);

            let kernel =
                MoeCountsOffsetsFusedKernel::new(&ctx).expect("fused kernel");
            let cb = ctx.command_queue.new_command_buffer();
            let args = MoeCountsOffsetsFusedArguments {
                topk_ids_buffer: &topk_ids_buf,
                counts_buffer: &counts_buf,
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

            // Verify counts
            let counts_ptr = counts_buf.contents() as *const u32;
            let counts_gpu =
                unsafe { std::slice::from_raw_parts(counts_ptr, e) };
            assert_eq!(
                counts_gpu,
                &counts_cpu[..],
                "counts mismatch T={} E={} K={}",
                t,
                e,
                k
            );

            // Verify offsets
            let offsets_ptr = offsets_buf.contents() as *const u32;
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
            let sum_ptr = sum_k_buf.contents() as *const u32;
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

    let counts_buf = alloc_buffer::<u32>(&ctx, e);
    let offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let sum_k_buf = alloc_buffer::<u32>(&ctx, 1);
    let num_tiles = ((e + 511) / 512).max(1);
    let partials_buf = alloc_buffer::<u32>(&ctx, num_tiles * 512);

    let kernel = MoeCountsOffsetsFusedKernel::new(&ctx).expect("fused kernel");
    let cb = ctx.command_queue.new_command_buffer();
    let args = MoeCountsOffsetsFusedArguments {
        topk_ids_buffer: &topk_ids_buf,
        counts_buffer: &counts_buf,
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

    let counts_gpu = unsafe {
        std::slice::from_raw_parts(counts_buf.contents() as *const u32, e)
    };
    let offsets_gpu = unsafe {
        std::slice::from_raw_parts(offsets_buf.contents() as *const u32, e + 1)
    };

    let mut expected_counts = vec![0u32; e];
    expected_counts[3] = (t * k) as u32;
    assert_eq!(counts_gpu, &expected_counts[..]);

    let mut expected_offsets = vec![0u32; e + 1];
    expected_offsets[4..].fill((t * k) as u32);
    assert_eq!(offsets_gpu, &expected_offsets[..]);

    // Case 2: T=0
    let (t, e, k) = (0usize, 8usize, 2usize);
    let topk_ids: Vec<i32> = vec![];
    let topk_ids_buf = alloc_buffer_with_data(&ctx, &topk_ids);
    let counts_buf = alloc_buffer::<u32>(&ctx, e);
    let offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
    let sum_k_buf = alloc_buffer::<u32>(&ctx, 1);
    let num_tiles = ((e + 511) / 512).max(1);
    let partials_buf = alloc_buffer::<u32>(&ctx, num_tiles * 512);

    let kernel = MoeCountsOffsetsFusedKernel::new(&ctx).expect("fused kernel");
    let cb = ctx.command_queue.new_command_buffer();
    let args = MoeCountsOffsetsFusedArguments {
        topk_ids_buffer: &topk_ids_buf,
        counts_buffer: &counts_buf,
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
        std::slice::from_raw_parts(offsets_buf.contents() as *const u32, e + 1)
    };
    assert!(offsets_gpu.iter().all(|&v| v == 0));
}
