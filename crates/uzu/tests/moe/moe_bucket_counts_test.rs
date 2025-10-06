#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::f16;
use metal::{Device, MTLResourceOptions};
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    MTLContext,
    kernel::{
        KernelDataType, MoeBucketCountsArguments, MoeBucketCountsKernel,
        MoeTopKArguments, MoeTopKKernel,
    },
};

fn create_test_context() -> Option<MTLContext> {
    let device = Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

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

fn gen_topk_ids_from_logits(
    ctx: &MTLContext,
    t: usize,
    e: usize,
    k: usize,
) -> (Vec<i32>, metal::Buffer) {
    let mut rng = StdRng::seed_from_u64(1234);
    let logits: Vec<f32> =
        (0..t * e).map(|_| rng.random_range(-4.0..4.0)).collect();

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
fn test_bucket_counts_parity_random() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping");
            return;
        },
    };
    let shapes = vec![(1usize, 4usize), (7, 16), (64, 64), (1024, 128)];
    let ks = vec![1usize, 2usize, 4usize];
    for &(t, e) in &shapes {
        for &k in &ks {
            if k > e {
                continue;
            }
            let (topk_ids, topk_ids_buf) =
                gen_topk_ids_from_logits(&ctx, t, e, k);
            let counts_cpu = cpu_bucket_counts(&topk_ids, t, k, e);

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

            let num_blocks = ((t + 255) / 256).max(1);
            let num_tiles = ((e + 512 - 1) / 512).max(1);
            let partials_buf = ctx.device.new_buffer(
                (num_blocks * num_tiles * 512 * std::mem::size_of::<u32>())
                    as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let kernel =
                MoeBucketCountsKernel::new(&ctx).expect("bucket kernel");
            let cb = ctx.command_queue.new_command_buffer();
            let args = MoeBucketCountsArguments {
                topk_ids_buffer: &topk_ids_buf,
                counts_buffer: &counts_buf,
                partials_buffer: &partials_buf,
                t,
                e,
                k,
            };
            kernel.encode(&cb, args).expect("encode bucket");
            cb.commit();
            cb.wait_until_completed();

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
        }
    }
}

#[test]
fn test_bucket_counts_edge_cases() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping");
            return;
        },
    };
    // All tokens to one expert
    let (t, e, k) = (16usize, 8usize, 2usize);
    let mut topk_ids = vec![-1i32; t * k];
    for ti in 0..t {
        for kk in 0..k {
            topk_ids[ti * k + kk] = 3;
        }
    }
    let topk_ids_buf = ctx.device.new_buffer_with_data(
        topk_ids.as_ptr() as *const _,
        (topk_ids.len() * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
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

    let num_blocks = ((t + 255) / 256).max(1);
    let num_tiles = ((e + 512 - 1) / 512).max(1);
    let partials_buf = ctx.device.new_buffer(
        (num_blocks * num_tiles * 512 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let kernel = MoeBucketCountsKernel::new(&ctx).expect("bucket kernel");
    let cb = ctx.command_queue.new_command_buffer();
    let args = MoeBucketCountsArguments {
        topk_ids_buffer: &topk_ids_buf,
        counts_buffer: &counts_buf,
        partials_buffer: &partials_buf,
        t,
        e,
        k,
    };
    kernel.encode(&cb, args).expect("encode bucket");
    cb.commit();
    cb.wait_until_completed();
    let counts_gpu = unsafe {
        std::slice::from_raw_parts(counts_buf.contents() as *const u32, e)
    };
    let mut expected = vec![0u32; e];
    expected[3] = (t * k) as u32;
    assert_eq!(counts_gpu, &expected[..]);

    // T=0
    let (t, e, k) = (0usize, 8usize, 2usize);
    let topk_ids: Vec<i32> = vec![];
    let topk_ids_buf = ctx.device.new_buffer_with_data(
        topk_ids.as_ptr() as *const _,
        0,
        MTLResourceOptions::StorageModeShared,
    );
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

    let num_blocks = ((t + 255) / 256).max(1);
    let num_tiles = ((e + 512 - 1) / 512).max(1);
    let partials_buf = ctx.device.new_buffer(
        (num_blocks * num_tiles * 512 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let kernel = MoeBucketCountsKernel::new(&ctx).expect("bucket kernel");
    let cb = ctx.command_queue.new_command_buffer();
    let args = MoeBucketCountsArguments {
        topk_ids_buffer: &topk_ids_buf,
        counts_buffer: &counts_buf,
        partials_buffer: &partials_buf,
        t,
        e,
        k,
    };
    kernel.encode(&cb, args).expect("encode bucket");
    cb.commit();
    cb.wait_until_completed();
    let counts_gpu = unsafe {
        std::slice::from_raw_parts(counts_buf.contents() as *const u32, e)
    };
    assert!(counts_gpu.iter().all(|&v| v == 0));

    // Inject out-of-range ids; expect ignore
    let (t, e, k) = (8usize, 8usize, 2usize);
    let mut topk_ids = vec![-1i32; t * k];
    for ti in 0..t {
        for kk in 0..k {
            topk_ids[ti * k + kk] = if (ti + kk) % 3 == 0 {
                100
            } else {
                -5
            };
        }
    }
    let topk_ids_buf = ctx.device.new_buffer_with_data(
        topk_ids.as_ptr() as *const _,
        (topk_ids.len() * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
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

    let num_blocks = ((t + 255) / 256).max(1);
    let num_tiles = ((e + 512 - 1) / 512).max(1);
    let partials_buf = ctx.device.new_buffer(
        (num_blocks * num_tiles * 512 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let kernel = MoeBucketCountsKernel::new(&ctx).expect("bucket kernel");
    let cb = ctx.command_queue.new_command_buffer();
    let args = MoeBucketCountsArguments {
        topk_ids_buffer: &topk_ids_buf,
        counts_buffer: &counts_buf,
        partials_buffer: &partials_buf,
        t,
        e,
        k,
    };
    kernel.encode(&cb, args).expect("encode bucket");
    cb.commit();
    cb.wait_until_completed();
    let counts_gpu = unsafe {
        std::slice::from_raw_parts(counts_buf.contents() as *const u32, e)
    };
    assert!(counts_gpu.iter().all(|&v| v == 0));
}
