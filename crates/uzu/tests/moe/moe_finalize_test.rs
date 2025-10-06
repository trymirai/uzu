#![cfg(feature = "moe_dev_tests")]

use half::f16;
use metal::MTLResourceOptions;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    MTLContext,
    kernel::{
        KernelDataType, MoeBlockBasesArguments, MoeBucketCountsArguments,
        MoeBucketCountsKernel, MoeFinalizeArguments, MoeFinalizeKernel,
        MoeOffsetsScanArguments, MoeOffsetsScanKernel, MoeScatterArguments,
        MoeScatterKernels, MoeScatterWithMapArguments, MoeTopKArguments,
        MoeTopKKernel,
    },
};

fn create_ctx() -> MTLContext {
    let device = metal::Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    MTLContext::new(device, queue).expect("ctx")
}

fn cpu_finalize(
    tok2row: &[i32],
    probs: &[f16],
    y_partial: &[f16],
    t: usize,
    d_model: usize,
    k: usize,
) -> Vec<f16> {
    let mut y = vec![f16::from_f32(0.0); t * d_model];
    for ti in 0..t {
        for f in 0..d_model {
            let mut acc = 0f32;
            for kk in 0..k {
                let idx = ti * k + kk;
                let row = tok2row[idx];
                if row >= 0 {
                    let rowu = row as usize;
                    acc += probs[idx].to_f32()
                        * y_partial[rowu * d_model + f].to_f32();
                }
            }
            y[ti * d_model + f] = f16::from_f32(acc);
        }
    }
    y
}

#[test]
fn test_moe_finalize_end_to_end() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(2026);
    let t = 37usize;
    let e = 17usize;
    let k = 2usize;
    let d_model = 96usize;
    let d_ff = 128usize;

    // Inputs
    let logits: Vec<f32> =
        (0..t * e).map(|_| rng.random_range(-4.0..4.0)).collect();

    // Buffers
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

    let topk = MoeTopKKernel::new(&ctx).expect("topk");
    let cb = ctx.command_queue.new_command_buffer();
    topk.encode(
        &cb,
        KernelDataType::Float32,
        MoeTopKArguments {
            logits_buffer: &logits_buf,
            topk_ids_buffer: &topk_ids_buf,
            topk_probs_buffer: &topk_probs_buf,
            t,
            e,
            k,
            renorm: true,
        },
    )
    .expect("encode topk");
    cb.commit();
    cb.wait_until_completed();

    // Bucket counts -> offsets -> sumk
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
    let bucket = MoeBucketCountsKernel::new(&ctx).expect("bucket");
    let cb = ctx.command_queue.new_command_buffer();
    bucket
        .encode(
            &cb,
            MoeBucketCountsArguments {
                partials_buffer: &partials_buf,
                topk_ids_buffer: &topk_ids_buf,
                counts_buffer: &counts_buf,
                t,
                e,
                k,
            },
        )
        .expect("encode bucket");
    cb.commit();
    cb.wait_until_completed();

    let offsets_buf = ctx.device.new_buffer(
        ((e + 1) * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sumk_buf = ctx.device.new_buffer(
        (1 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let scan = MoeOffsetsScanKernel::new(&ctx).expect("scan");
    let cb = ctx.command_queue.new_command_buffer();
    scan.encode(
        &cb,
        MoeOffsetsScanArguments {
            counts_buffer: &counts_buf,
            offsets_buffer: &offsets_buf,
            sumk_buffer: &sumk_buf,
            e,
        },
    )
    .expect("encode scan");
    cb.commit();
    cb.wait_until_completed();

    let sumk = unsafe { *(sumk_buf.contents() as *const u32) } as usize;
    let out_ids_buf = ctx.device.new_buffer(
        (sumk * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_probs_buf = ctx.device.new_buffer(
        (sumk * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Partials (Pass A) to get block_bases
    let num_blocks = (t + 256 - 1) / 256;
    let num_tiles = (e + 512 - 1) / 512;
    let entries = num_blocks * num_tiles * 512usize;
    let partials = ctx.device.new_buffer(
        (entries * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_partials = ctx
        .compute_pipeline_state("moe_bucket_partials", None)
        .expect("partials");
    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&k_partials);
    enc.set_buffer(0, Some(&topk_ids_buf), 0);
    enc.set_buffer(1, Some(&partials), 0);
    let t_u = t as u32;
    let e_u = e as u32;
    let k_u = k as u32;
    let nb_u = num_blocks as u32;
    enc.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &t_u as *const u32 as *const _,
    );
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &e_u as *const u32 as *const _,
    );
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &k_u as *const u32 as *const _,
    );
    enc.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &nb_u as *const u32 as *const _,
    );
    let tg = metal::MTLSize::new(num_blocks as u64, 1, 1);
    let tpt = metal::MTLSize::new(256, 1, 1);
    enc.dispatch_thread_groups(tg, tpt);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    // Block bases
    let block_bases_buf = ctx.device.new_buffer(
        (entries * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let scatter = MoeScatterKernels::new(&ctx).expect("scatter kernels");
    let cb = ctx.command_queue.new_command_buffer();
    let block_alloc_buf = ctx.device.new_buffer(
        (num_blocks * num_tiles * 512 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    scatter
        .encode_block_bases(
            &cb,
            MoeBlockBasesArguments {
                partials_buffer: &partials,
                block_bases_buffer: &block_bases_buf,
                block_alloc_buffer: &block_alloc_buf,
                e,
                num_blocks,
                num_tiles,
            },
        )
        .expect("encode bases");
    cb.commit();
    cb.wait_until_completed();

    // tok2row map buffer
    let tok2row_len = t * k;
    let tok2row_buf = ctx.device.new_buffer(
        (tok2row_len * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::write_bytes(
            tok2row_buf.contents(),
            0xff,
            tok2row_len * std::mem::size_of::<i32>(),
        );
    }

    // Scatter with map
    let cb = ctx.command_queue.new_command_buffer();
    scatter
        .encode_scatter_with_map(
            &cb,
            MoeScatterWithMapArguments {
                base: MoeScatterArguments {
                    topk_ids_buffer: &topk_ids_buf,
                    topk_probs_buffer: &topk_probs_buf,
                    offsets_buffer: &offsets_buf,
                    block_bases_buffer: &block_bases_buf,
                    block_alloc_buffer: &block_alloc_buf,
                    out_ids_buffer: &out_ids_buf,
                    out_probs_buffer: &out_probs_buf,
                    t,
                    e,
                    k,
                    num_blocks,
                    num_tiles,
                },
                tok2row_buffer: &tok2row_buf,
            },
            KernelDataType::Float16,
        )
        .expect("encode scatter map");
    cb.commit();
    cb.wait_until_completed();

    // Finalize
    let y_partial_buf = ctx.device.new_buffer(
        (sumk * d_model * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mut y_partial = vec![f16::from_f32(0.0); sumk * d_model];
    for item in &mut y_partial {
        *item = f16::from_f32(rng.random_range(-2.0..2.0));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            y_partial.as_ptr(),
            y_partial_buf.contents() as *mut f16,
            y_partial.len(),
        );
    }

    let y_out_buf = ctx.device.new_buffer(
        (t * d_model * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let finalize = MoeFinalizeKernel::new(&ctx).expect("finalize");
    let cb = ctx.command_queue.new_command_buffer();
    finalize
        .encode(
            &cb,
            MoeFinalizeArguments {
                tok2row_buffer: &tok2row_buf,
                probs_buffer: &topk_probs_buf,
                y_partial_buffer: &y_partial_buf,
                y_out_buffer: &y_out_buf,
                t,
                d_model,
                k,
            },
            KernelDataType::Float16,
        )
        .expect("encode finalize");
    cb.commit();
    cb.wait_until_completed();

    // Compare with CPU
    let tok2row = unsafe {
        std::slice::from_raw_parts(tok2row_buf.contents() as *const i32, t * k)
    };
    let probs_h = unsafe {
        std::slice::from_raw_parts(
            topk_probs_buf.contents() as *const f16,
            t * k,
        )
    };
    let y_partial_h = unsafe {
        std::slice::from_raw_parts(
            y_partial_buf.contents() as *const f16,
            sumk * d_model,
        )
    };
    let y_out_h = unsafe {
        std::slice::from_raw_parts(
            y_out_buf.contents() as *const f16,
            t * d_model,
        )
    };
    let y_cpu = cpu_finalize(tok2row, probs_h, y_partial_h, t, d_model, k);
    for i in 0..(t * d_model) {
        let a = y_out_h[i].to_f32();
        let b = y_cpu[i].to_f32();
        assert!((a - b).abs() < 1e-2, "mismatch at {}: {} vs {}", i, a, b);
    }
}
