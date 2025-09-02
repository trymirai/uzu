#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::f16;
use metal::{Device, MTLResourceOptions};
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    MTLContext,
    kernel::{
        KernelDataType, MoeBucketCountsArguments, MoeBucketCountsKernel,
        MoeOffsetsScanArguments, MoeOffsetsScanKernel, MoeTopKArguments,
        MoeTopKKernel,
    },
};

fn create_test_context() -> Option<MTLContext> {
    let device = Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

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
    let logits_buf = ctx.device.new_buffer_with_data(
        logits.as_ptr() as *const _,
        (logits.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let topk_ids_buf = ctx.device.new_buffer(
        (t * k * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let topk_probs_buf = ctx.device.new_buffer(
        (t * k * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let topk = MoeTopKKernel::new(ctx).expect("topk");
    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let args = MoeTopKArguments {
        logits_buffer: &logits_buf,
        topk_ids_buffer: &topk_ids_buf,
        topk_probs_buffer: &topk_probs_buf,
        t,
        e,
        k,
        renorm: true,
    };
    topk.encode(&enc, KernelDataType::Float32, args).expect("encode topk");
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let counts_buf = ctx.device.new_buffer(
        (e * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::write_bytes(
            counts_buf.contents(),
            0,
            e * std::mem::size_of::<u32>(),
        );
    }
    let bucket = MoeBucketCountsKernel::new(ctx).expect("bucket");
    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let bargs = MoeBucketCountsArguments {
        topk_ids_buffer: &topk_ids_buf,
        counts_buffer: &counts_buf,
        t,
        e,
        k,
    };
    bucket.encode(&enc, bargs).expect("encode bucket");
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    counts_buf
}

#[test]
fn test_offsets_scan_parity() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping");
            return;
        },
    };
    let shapes =
        vec![(1usize, 4usize, 1usize), (7, 16, 2), (64, 64, 2), (128, 128, 4)];
    for (t, e, k) in shapes {
        let counts_buf = gen_counts_via_gpu(&ctx, t, e, k);
        let counts = unsafe {
            std::slice::from_raw_parts(counts_buf.contents() as *const u32, e)
        };
        let (offsets_cpu, sumk_cpu) = cpu_exclusive_scan(counts);

        let offsets_buf = ctx.device.new_buffer(
            ((e + 1) * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let sumk_buf = ctx.device.new_buffer(
            (1 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let kernel = MoeOffsetsScanKernel::new(&ctx).expect("scan");
        let cb = ctx.command_queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let args = MoeOffsetsScanArguments {
            counts_buffer: &counts_buf,
            offsets_buffer: &offsets_buf,
            sumk_buffer: &sumk_buf,
            e,
        };
        kernel.encode(&enc, args).expect("encode scan");
        enc.end_encoding();
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
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping");
            return;
        },
    };

    // E=0
    let counts_buf =
        ctx.device.new_buffer(0, MTLResourceOptions::StorageModeShared);
    let offsets_buf = ctx.device.new_buffer(
        (1 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sumk_buf = ctx.device.new_buffer(
        (1 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let kernel = MoeOffsetsScanKernel::new(&ctx).expect("scan");
    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let args = MoeOffsetsScanArguments {
        counts_buffer: &counts_buf,
        offsets_buffer: &offsets_buf,
        sumk_buffer: &sumk_buf,
        e: 0,
    };
    kernel.encode(&enc, args).expect("encode scan");
    enc.end_encoding();
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
