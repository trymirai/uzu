#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::f16;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    MTLContext,
    kernel::{
        KernelDataType, MoeBucketCountsArguments, MoeBucketCountsKernel,
        MoeOffsetsScanArguments, MoeOffsetsScanKernel, MoeTopKArguments,
        MoeTopKKernel,
    },
};

use super::test_utils::{alloc_buffer, alloc_buffer_with_data, create_ctx};

fn cpu_exclusive_scan(counts: &[u32]) -> (Vec<u32>, u32) {
    let mut offsets: Vec<u32> = Vec::with_capacity(counts.len() + 1);
    offsets.push(0u32);
    for i in 0..counts.len() {
        let next: u32 = offsets[i].wrapping_add(counts[i]);
        offsets.push(next);
    }
    let sumk: u32 = *offsets.last().unwrap_or(&0u32);
    (offsets, sumk)
}

fn gen_counts_via_gpu(
    ctx: &MTLContext,
    t: usize,
    e: usize,
    k: usize,
) -> metal::Buffer {
    // Generate logits -> topk_ids -> counts
    let mut rng = StdRng::seed_from_u64(2025);
    let logits: Vec<f32> =
        (0..t * e).map(|_| rng.random_range(-3.0..3.0)).collect();
    let logits_buf = alloc_buffer_with_data(&ctx, &logits);
    let topk_ids_buf = alloc_buffer::<i32>(&ctx, t * k);
    let topk_probs_buf = alloc_buffer::<f16>(&ctx, t * k);
    let topk = MoeTopKKernel::new(ctx).expect("topk");
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

    let counts_buf = alloc_buffer::<u32>(&ctx, e);
    unsafe {
        std::ptr::write_bytes(
            counts_buf.contents(),
            0,
            e * std::mem::size_of::<u32>(),
        );
    }
    let num_blocks = ((t + 255) / 256).max(1);
    let num_tiles = ((e + 512 - 1) / 512).max(1);
    let partials_buf = alloc_buffer::<u32>(&ctx, num_blocks * num_tiles * 512);
    let bucket = MoeBucketCountsKernel::new(ctx).expect("bucket");
    let cb = ctx.command_queue.new_command_buffer();
    let bargs = MoeBucketCountsArguments {
        partials_buffer: &partials_buf,
        topk_ids_buffer: &topk_ids_buf,
        counts_buffer: &counts_buf,
        t,
        e,
        k,
    };
    bucket.encode(&cb, bargs).expect("encode bucket");
    cb.commit();
    cb.wait_until_completed();
    counts_buf
}

#[test]
fn test_offsets_scan_parity() {
    let ctx = create_ctx();
    let shapes =
        vec![(1usize, 4usize, 1usize), (7, 16, 2), (64, 64, 2), (128, 128, 4)];
    for (t, e, k) in shapes {
        let counts_buf = gen_counts_via_gpu(&ctx, t, e, k);
        let counts = unsafe {
            std::slice::from_raw_parts(counts_buf.contents() as *const u32, e)
        };
        let (offsets_cpu, sumk_cpu) = cpu_exclusive_scan(counts);

        let offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
        let sumk_buf = alloc_buffer::<u32>(&ctx, 1);

        let kernel = MoeOffsetsScanKernel::new(&ctx).expect("scan");
        let cb = ctx.command_queue.new_command_buffer();
        let args = MoeOffsetsScanArguments {
            counts_buffer: &counts_buf,
            offsets_buffer: &offsets_buf,
            sumk_buffer: &sumk_buf,
            e,
        };
        kernel.encode(&cb, args).expect("encode scan");
        cb.commit();
        cb.wait_until_completed();

        let offsets_gpu = unsafe {
            std::slice::from_raw_parts(
                offsets_buf.contents() as *const u32,
                e + 1,
            )
        };
        let sumk_gpu = unsafe {
            std::slice::from_raw_parts(sumk_buf.contents() as *const u32, 1)
        }[0];
        assert_eq!(offsets_gpu, &offsets_cpu[..], "offsets mismatch E={}", e);
        assert_eq!(sumk_gpu, sumk_cpu, "sum_k mismatch E={}", e);
    }
}

#[test]
fn test_offsets_scan_edge_cases() {
    let ctx = create_ctx();

    // E=0
    let counts_buf = alloc_buffer::<u32>(&ctx, 0);
    let offsets_buf = alloc_buffer::<u32>(&ctx, 1);
    let sumk_buf = alloc_buffer::<u32>(&ctx, 1);
    let kernel = MoeOffsetsScanKernel::new(&ctx).expect("scan");
    let cb = ctx.command_queue.new_command_buffer();
    let args = MoeOffsetsScanArguments {
        counts_buffer: &counts_buf,
        offsets_buffer: &offsets_buf,
        sumk_buffer: &sumk_buf,
        e: 0,
    };
    kernel.encode(&cb, args).expect("encode scan");
    cb.commit();
    cb.wait_until_completed();
    let offsets_gpu = unsafe {
        std::slice::from_raw_parts(offsets_buf.contents() as *const u32, 1)
    };
    let sumk_gpu = unsafe {
        std::slice::from_raw_parts(sumk_buf.contents() as *const u32, 1)
    }[0];
    assert_eq!(offsets_gpu[0], 0);
    assert_eq!(sumk_gpu, 0);
}
