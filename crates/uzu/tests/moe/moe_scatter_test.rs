#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::f16;
use metal::{Device, MTLResourceOptions};
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::{
    MTLContext,
    kernel::{
        KernelDataType, MoeBlockBasesArguments, MoeBucketCountsArguments,
        MoeBucketCountsKernel, MoeOffsetsScanArguments, MoeOffsetsScanKernel,
        MoeScatterArguments, MoeScatterKernels, MoeTopKArguments,
        MoeTopKKernel,
    },
};

use super::test_utils::create_ctx;

fn create_test_context() -> Option<MTLContext> {
    let device = Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

fn cpu_expert_buckets(
    topk_ids: &[i32],
    topk_probs: &[f32],
    t: usize,
    e: usize,
    k: usize,
) -> (Vec<i32>, Vec<f32>, Vec<u32>) {
    let mut per_e: Vec<Vec<(i32, f32)>> = vec![Vec::new(); e];
    for ti in 0..t {
        for kk in 0..k {
            let id = topk_ids[ti * k + kk];
            if id >= 0 {
                let ue = id as usize;
                if ue < e {
                    per_e[ue].push((ti as i32, topk_probs[ti * k + kk]));
                }
            }
        }
    }
    let mut offsets = Vec::with_capacity(e + 1);
    offsets.push(0u32);
    for ei in 0..e {
        offsets.push(offsets[ei] + per_e[ei].len() as u32);
    }
    let sumk = *offsets.last().unwrap() as usize;
    let mut ids = vec![0i32; sumk];
    let mut probs = vec![0f32; sumk];
    for ei in 0..e {
        let mut entries = per_e[ei].clone();
        entries.sort_by_key(|&(id, _)| id);
        let start = offsets[ei] as usize;
        for (i, (id, p)) in entries.into_iter().enumerate() {
            ids[start + i] = id;
            probs[start + i] = p;
        }
    }
    (ids, probs, offsets)
}

#[test]
fn test_scatter_buckets_parity() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping");
            return;
        },
    };
    let shapes = vec![(1usize, 4usize, 1usize), (7, 16, 2), (128, 64, 2)];
    for (t, e, k) in shapes {
        let mut rng = StdRng::seed_from_u64(7);
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

        // Read back CPU probs for reference compare
        let probs_f16 = unsafe {
            std::slice::from_raw_parts(
                topk_probs_buf.contents() as *const f16,
                t * k,
            )
        };
        let topk_probs_cpu: Vec<f32> =
            probs_f16.iter().map(|&h| h.to_f32()).collect();
        let topk_ids_cpu = unsafe {
            std::slice::from_raw_parts(
                topk_ids_buf.contents() as *const i32,
                t * k,
            )
        };

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

        // Build real partials via counts Pass-A
        let (partials_buf, num_blocks, num_tiles) =
            run_partials(&ctx, &topk_ids_buf, t, e, k);
        let entries = num_blocks * num_tiles * 512usize;
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
                    partials_buffer: &partials_buf,
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

        let sumk = unsafe {
            std::slice::from_raw_parts(sumk_buf.contents() as *const u32, 1)
        }[0] as usize;
        let out_ids_buf = ctx.device.new_buffer(
            (sumk * std::mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_probs_buf = ctx.device.new_buffer(
            (sumk * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cb = ctx.command_queue.new_command_buffer();
        scatter
            .encode_scatter(
                &cb,
                MoeScatterArguments {
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
                KernelDataType::Float16,
            )
            .expect("encode scatter");
        cb.commit();
        cb.wait_until_completed();

        let out_ids = unsafe {
            std::slice::from_raw_parts(
                out_ids_buf.contents() as *const i32,
                sumk,
            )
        };
        let out_probs_h = unsafe {
            std::slice::from_raw_parts(
                out_probs_buf.contents() as *const f16,
                sumk,
            )
        };
        let _out_probs: Vec<f32> =
            out_probs_h.iter().map(|&h| h.to_f32()).collect();

        // CPU reference (stable per expert)
        let (cpu_ids, _cpu_probs, offsets_cpu) =
            cpu_expert_buckets(topk_ids_cpu, &topk_probs_cpu, t, e, k);
        let offsets_gpu = unsafe {
            std::slice::from_raw_parts(
                offsets_buf.contents() as *const u32,
                e + 1,
            )
        };
        assert_eq!(offsets_gpu, &offsets_cpu[..]);

        // Compare per-expert multisets of ids
        for ei in 0..e {
            let s = offsets_cpu[ei] as usize;
            let epos = offsets_cpu[ei + 1] as usize;
            let mut a = out_ids[s..epos].to_vec();
            a.sort();
            let mut b = cpu_ids[s..epos].to_vec();
            b.sort();
            assert_eq!(a, b, "ids multiset mismatch for expert {}", ei);
        }
    }
}

const BLOCK_SIZE: usize = 256;
const TILE_E: usize = 512;

fn compile(
    ctx: &MTLContext,
    name: &str,
) -> metal::ComputePipelineState {
    ctx.compute_pipeline_state(name, None).expect(name)
}

fn run_partials(
    ctx: &MTLContext,
    topk_ids_buf: &metal::Buffer,
    t: usize,
    e: usize,
    k: usize,
) -> (metal::Buffer, usize, usize) {
    let num_blocks = (t + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let num_tiles = (e + TILE_E - 1) / TILE_E;
    let entries = num_blocks * num_tiles * TILE_E;
    let partials = ctx.device.new_buffer(
        (entries * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let k_partials = compile(ctx, "moe_bucket_partials");
    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&k_partials);
    enc.set_buffer(0, Some(topk_ids_buf), 0);
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
    let tpt = metal::MTLSize::new(BLOCK_SIZE as u64, 1, 1);
    enc.dispatch_thread_groups(tg, tpt);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    (partials, num_blocks, num_tiles)
}

fn run_counts_reduce(
    ctx: &MTLContext,
    partials: &metal::Buffer,
    e: usize,
    num_blocks: usize,
) -> metal::Buffer {
    let k_reduce = compile(ctx, "moe_bucket_reduce_partials");
    let counts = ctx.device.new_buffer(
        (e * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&k_reduce);
    enc.set_buffer(0, Some(partials), 0);
    enc.set_buffer(1, Some(&counts), 0);
    let e_u = e as u32;
    let nb_u = num_blocks as u32;
    enc.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &e_u as *const u32 as *const _,
    );
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &nb_u as *const u32 as *const _,
    );
    let tg = metal::MTLSize::new(((e + 255) / 256) as u64, 1, 1);
    let tpt = metal::MTLSize::new(256, 1, 1);
    enc.dispatch_thread_groups(tg, tpt);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    counts
}

fn run_offsets(
    ctx: &MTLContext,
    counts: &metal::Buffer,
    e: usize,
) -> (metal::Buffer, metal::Buffer) {
    let offsets = ctx.device.new_buffer(
        ((e + 1) * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sumk = ctx.device.new_buffer(
        (1 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let scan =
        uzu::backends::metal::kernel::MoeOffsetsScanKernel::new(ctx).unwrap();
    let cb = ctx.command_queue.new_command_buffer();
    scan.encode(
        &cb,
        uzu::backends::metal::kernel::MoeOffsetsScanArguments {
            counts_buffer: counts,
            offsets_buffer: &offsets,
            sumk_buffer: &sumk,
            e,
        },
    )
    .unwrap();
    cb.commit();
    cb.wait_until_completed();
    (offsets, sumk)
}

fn run_block_bases(
    ctx: &MTLContext,
    partials: &metal::Buffer,
    e: usize,
    num_blocks: usize,
    num_tiles: usize,
    capacity: u32,
) -> (metal::Buffer, metal::Buffer) {
    let k_bases = compile(ctx, "moe_block_bases_from_partials");
    let entries = num_blocks * num_tiles * TILE_E;
    let bases = ctx.device.new_buffer(
        (entries * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let alloc = ctx.device.new_buffer(
        (entries * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&k_bases);
    enc.set_buffer(0, Some(partials), 0);
    enc.set_buffer(1, Some(&bases), 0);
    enc.set_buffer(2, Some(&alloc), 0);
    let e_u = e as u32;
    let nb_u = num_blocks as u32;
    let nt_u = num_tiles as u32;
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &e_u as *const u32 as *const _,
    );
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &nb_u as *const u32 as *const _,
    );
    enc.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &nt_u as *const u32 as *const _,
    );
    enc.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        &capacity as *const u32 as *const _,
    );
    let total_entries = num_tiles * TILE_E;
    let tg = metal::MTLSize::new(((total_entries + 255) / 256) as u64, 1, 1);
    let tpt = metal::MTLSize::new(256, 1, 1);
    enc.dispatch_thread_groups(tg, tpt);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    (bases, alloc)
}

fn run_scatter(
    ctx: &MTLContext,
    topk_ids: &metal::Buffer,
    topk_probs: &metal::Buffer,
    offsets: &metal::Buffer,
    bases: &metal::Buffer,
    alloc: &metal::Buffer,
    t: usize,
    e: usize,
    k: usize,
    num_blocks: usize,
    num_tiles: usize,
    sumk: usize,
) -> (metal::Buffer, metal::Buffer) {
    let k_scatter = compile(ctx, "moe_scatter_buckets_f16");
    let out_ids = ctx.device.new_buffer(
        (sumk * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_probs = ctx.device.new_buffer(
        (sumk * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&k_scatter);
    enc.set_buffer(0, Some(topk_ids), 0);
    enc.set_buffer(1, Some(topk_probs), 0);
    enc.set_buffer(2, Some(offsets), 0);
    enc.set_buffer(3, Some(bases), 0);
    enc.set_buffer(4, Some(alloc), 0);
    enc.set_buffer(5, Some(&out_ids), 0);
    enc.set_buffer(6, Some(&out_probs), 0);
    let t_u = t as u32;
    let e_u = e as u32;
    let k_u = k as u32;
    let nb_u = num_blocks as u32;
    let nt_u = num_tiles as u32;
    enc.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        &t_u as *const u32 as *const _,
    );
    enc.set_bytes(
        8,
        std::mem::size_of::<u32>() as u64,
        &e_u as *const u32 as *const _,
    );
    enc.set_bytes(
        9,
        std::mem::size_of::<u32>() as u64,
        &k_u as *const u32 as *const _,
    );
    enc.set_bytes(
        10,
        std::mem::size_of::<u32>() as u64,
        &nb_u as *const u32 as *const _,
    );
    enc.set_bytes(
        11,
        std::mem::size_of::<u32>() as u64,
        &nt_u as *const u32 as *const _,
    );
    let tg = metal::MTLSize::new(num_blocks as u64, 1, 1);
    let tpt = metal::MTLSize::new(BLOCK_SIZE as u64, 1, 1);
    enc.dispatch_thread_groups(tg, tpt);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    (out_ids, out_probs)
}

fn build_cpu_multisets(
    topk_ids: &[i32],
    t: usize,
    e: usize,
    k: usize,
) -> Vec<Vec<i32>> {
    let mut per_e: Vec<Vec<i32>> = vec![Vec::new(); e];
    for ti in 0..t {
        for kk in 0..k {
            let id = topk_ids[ti * k + kk];
            if id >= 0 {
                let ue = id as usize;
                if ue < e {
                    per_e[ue].push(ti as i32);
                }
            }
        }
    }
    per_e
}

#[test]
fn test_multiblock_multitile_parity_real_partials() {
    let ctx = create_ctx();
    let t = 6 * BLOCK_SIZE;
    let e = 2 * TILE_E + 37;
    for &k in &[1usize, 2, 4] {
        let mut rng = StdRng::seed_from_u64(123);
        let logits: Vec<f32> =
            (0..t * e).map(|_| rng.random_range(-2.0..2.0)).collect();
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
        let topk = MoeTopKKernel::new(&ctx).unwrap();
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
        .unwrap();
        cb.commit();
        cb.wait_until_completed();

        let (partials, num_blocks, num_tiles) =
            run_partials(&ctx, &topk_ids_buf, t, e, k);
        let counts_buf = run_counts_reduce(&ctx, &partials, e, num_blocks);
        let (offsets_buf, sumk_buf) = run_offsets(&ctx, &counts_buf, e);
        let sumk = unsafe { *(sumk_buf.contents() as *const u32) } as usize;
        let counts = unsafe {
            std::slice::from_raw_parts(counts_buf.contents() as *const u32, e)
        };
        assert_eq!(sumk as u32, counts.iter().copied().sum::<u32>());

        let (bases, alloc) =
            run_block_bases(&ctx, &partials, e, num_blocks, num_tiles, 0);
        let (out_ids_buf, _out_probs_buf) = run_scatter(
            &ctx,
            &topk_ids_buf,
            &topk_probs_buf,
            &offsets_buf,
            &bases,
            &alloc,
            t,
            e,
            k,
            num_blocks,
            num_tiles,
            sumk,
        );

        let cpu_sets = build_cpu_multisets(
            unsafe {
                std::slice::from_raw_parts(
                    topk_ids_buf.contents() as *const i32,
                    t * k,
                )
            },
            t,
            e,
            k,
        );
        let offsets = unsafe {
            std::slice::from_raw_parts(
                offsets_buf.contents() as *const u32,
                e + 1,
            )
        };
        let out_ids = unsafe {
            std::slice::from_raw_parts(
                out_ids_buf.contents() as *const i32,
                sumk,
            )
        };

        // per expert multiset
        for ei in 0..e {
            let s = offsets[ei] as usize;
            let epos = offsets[ei + 1] as usize;
            let mut gpu = out_ids[s..epos].to_vec();
            gpu.sort();
            let mut cpu = cpu_sets[ei].clone();
            cpu.sort();
            assert_eq!(gpu, cpu, "expert {}", ei);
        }
        // no OOB
        assert!(offsets[e] as usize == sumk);
    }
}

#[test]
fn test_capacity_clamp_correctness() {
    let ctx = create_ctx();
    let t = 5 * BLOCK_SIZE;
    let e = TILE_E + 5;
    let k = 2usize;
    // Skewed routing logits to drive most to e=0
    let mut rng = StdRng::seed_from_u64(77);
    let mut logits: Vec<f32> = vec![0.0; t * e];
    for ti in 0..t {
        for eid in 0..e {
            logits[ti * e + eid] = if eid == 0 {
                3.0 + rng.random_range(0.0..0.1)
            } else {
                rng.random_range(-3.0..-1.0)
            };
        }
    }
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
    let topk = MoeTopKKernel::new(&ctx).unwrap();
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
    .unwrap();
    cb.commit();
    cb.wait_until_completed();

    let (partials, num_blocks, num_tiles) =
        run_partials(&ctx, &topk_ids_buf, t, e, k);
    // CPU raw counts
    let topk_ids = unsafe {
        std::slice::from_raw_parts(topk_ids_buf.contents() as *const i32, t * k)
    };
    let mut raw_counts = vec![0u32; e];
    for ti in 0..t {
        for kk in 0..k {
            let id = topk_ids[ti * k + kk];
            if id >= 0 {
                let ue = id as usize;
                if ue < e {
                    raw_counts[ue] += 1;
                }
            }
        }
    }
    let avg = (raw_counts.iter().sum::<u32>() + e as u32 - 1) / e as u32;
    let capacity = (avg as f32 * 0.75) as u32; // simple uniform cap per expert
    let (bases, alloc) =
        run_block_bases(&ctx, &partials, e, num_blocks, num_tiles, capacity);
    // Build clamped counts by summing alloc per expert
    let alloc_ptr = alloc.contents() as *const u32;
    let alloc_slice = unsafe {
        std::slice::from_raw_parts(alloc_ptr, num_blocks * num_tiles * TILE_E)
    };
    let mut clamped_counts = vec![0u32; e];
    for tile_id in 0..num_tiles {
        for te in 0..TILE_E {
            let eid = tile_id * TILE_E + te;
            if eid >= e {
                break;
            }
            let mut sum = 0u32;
            for b in 0..num_blocks {
                let idx = (b * num_tiles + tile_id) * TILE_E + te;
                sum += alloc_slice[idx];
            }
            clamped_counts[eid] = sum;
        }
    }
    let counts_buf = ctx.device.new_buffer_with_data(
        clamped_counts.as_ptr() as *const _,
        (e * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (offsets_buf, sumk_buf) = run_offsets(&ctx, &counts_buf, e);
    let sumk = unsafe { *(sumk_buf.contents() as *const u32) } as usize;
    assert_eq!(sumk as u32, clamped_counts.iter().copied().sum::<u32>());

    let (_out_ids_buf, _out_probs_buf) = run_scatter(
        &ctx,
        &topk_ids_buf,
        &topk_probs_buf,
        &offsets_buf,
        &bases,
        &alloc,
        t,
        e,
        k,
        num_blocks,
        num_tiles,
        sumk,
    );
    let offsets = unsafe {
        std::slice::from_raw_parts(offsets_buf.contents() as *const u32, e + 1)
    };
    for ei in 0..e {
        let seg_len = offsets[ei + 1] - offsets[ei];
        assert_eq!(seg_len, std::cmp::min(raw_counts[ei], capacity));
    }
    // no writes beyond segment end
    assert_eq!(offsets[e] as usize, sumk);
    if e > 0 {
        assert!(offsets[0] == 0);
    }
    // Deterministic block precedence implicitly verified via clamped allocations per block ordering
}

#[test]
fn test_content_prob_pairing_integrity() {
    let ctx = create_ctx();
    let t = 4 * BLOCK_SIZE;
    let e = TILE_E + 11;
    let k = 2usize;
    let mut rng = StdRng::seed_from_u64(2025);
    let logits: Vec<f32> =
        (0..t * e).map(|_| rng.random_range(-2.0..2.0)).collect();
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
    let topk = MoeTopKKernel::new(&ctx).unwrap();
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
    .unwrap();
    cb.commit();
    cb.wait_until_completed();

    let (partials, num_blocks, num_tiles) =
        run_partials(&ctx, &topk_ids_buf, t, e, k);
    let counts_buf = run_counts_reduce(&ctx, &partials, e, num_blocks);
    let (offsets_buf, sumk_buf) = run_offsets(&ctx, &counts_buf, e);
    let (bases, alloc) =
        run_block_bases(&ctx, &partials, e, num_blocks, num_tiles, 0);
    let sumk = unsafe { *(sumk_buf.contents() as *const u32) } as usize;
    let (out_ids_buf, out_probs_buf) = run_scatter(
        &ctx,
        &topk_ids_buf,
        &topk_probs_buf,
        &offsets_buf,
        &bases,
        &alloc,
        t,
        e,
        k,
        num_blocks,
        num_tiles,
        sumk,
    );
    let out_ids = unsafe {
        std::slice::from_raw_parts(out_ids_buf.contents() as *const i32, sumk)
    };
    let out_probs_h = unsafe {
        std::slice::from_raw_parts(out_probs_buf.contents() as *const f16, sumk)
    };
    let out_probs: Vec<f32> = out_probs_h.iter().map(|&h| h.to_f32()).collect();
    let topk_ids = unsafe {
        std::slice::from_raw_parts(topk_ids_buf.contents() as *const i32, t * k)
    };
    let topk_probs_h = unsafe {
        std::slice::from_raw_parts(
            topk_probs_buf.contents() as *const f16,
            t * k,
        )
    };
    let topk_probs_cpu: Vec<f32> =
        topk_probs_h.iter().map(|&h| h.to_f32()).collect();
    let offsets = unsafe {
        std::slice::from_raw_parts(offsets_buf.contents() as *const u32, e + 1)
    };

    // Build CPU map (token, expert)->prob from topk
    let mut cpu_map: Vec<Vec<(i32, f32)>> = vec![Vec::new(); e];
    for ti in 0..t {
        for kk in 0..k {
            let id = topk_ids[ti * k + kk];
            if id >= 0 {
                let ue = id as usize;
                if ue < e {
                    cpu_map[ue].push((ti as i32, topk_probs_cpu[ti * k + kk]));
                }
            }
        }
    }

    for ei in 0..e {
        let s = offsets[ei] as usize;
        let epos = offsets[ei + 1] as usize;
        let mut v = cpu_map[ei].clone();
        v.sort_by_key(|&(tid, _)| tid);
        let mut gpu: Vec<(i32, f32)> =
            (s..epos).map(|i| (out_ids[i], out_probs[i])).collect();
        gpu.sort_by_key(|&(tid, _)| tid);
        assert_eq!(v.len(), gpu.len());
        for j in 0..v.len() {
            let (ct, cp) = v[j];
            let (gt, gp) = gpu[j];
            assert_eq!(ct, gt);
            assert!((cp - gp).abs() <= 1e-3f32);
        }
    }
}

#[test]
fn test_adversarial_ids_and_determinism() {
    let ctx = create_ctx();
    let t = 2 * BLOCK_SIZE;
    let e = TILE_E + 3;
    let k = 4usize;
    // Craft topk ids: negatives and out-of-range
    let mut topk_ids: Vec<i32> = vec![0; t * k];
    let mut topk_probs: Vec<f16> = vec![f16::from_f32(0.0); t * k];
    for ti in 0..t {
        for kk in 0..k {
            let id = match (ti + kk) % 5 {
                0 => -1,
                1 => e as i32,
                2 => 0,
                3 => (e - 1) as i32,
                _ => (ti % std::cmp::min(e, 20)) as i32,
            };
            topk_ids[ti * k + kk] = id;
            topk_probs[ti * k + kk] =
                f16::from_f32(0.1 * (kk as f32) + 0.001 * (ti as f32));
        }
    }
    let topk_ids_buf = ctx.device.new_buffer_with_data(
        topk_ids.as_ptr() as *const _,
        (topk_ids.len() * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let topk_probs_buf = ctx.device.new_buffer_with_data(
        topk_probs.as_ptr() as *const _,
        (topk_probs.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (partials, num_blocks, num_tiles) =
        run_partials(&ctx, &topk_ids_buf, t, e, k);
    let counts_buf = run_counts_reduce(&ctx, &partials, e, num_blocks);
    let (offsets_buf, sumk_buf) = run_offsets(&ctx, &counts_buf, e);
    let (bases, alloc) =
        run_block_bases(&ctx, &partials, e, num_blocks, num_tiles, 0);
    let sumk = unsafe { *(sumk_buf.contents() as *const u32) } as usize;
    let (out_ids_a, out_probs_a) = run_scatter(
        &ctx,
        &topk_ids_buf,
        &topk_probs_buf,
        &offsets_buf,
        &bases,
        &alloc,
        t,
        e,
        k,
        num_blocks,
        num_tiles,
        sumk,
    );
    let (out_ids_b, out_probs_b) = run_scatter(
        &ctx,
        &topk_ids_buf,
        &topk_probs_buf,
        &offsets_buf,
        &bases,
        &alloc,
        t,
        e,
        k,
        num_blocks,
        num_tiles,
        sumk,
    );
    let a_ids = unsafe {
        std::slice::from_raw_parts(out_ids_a.contents() as *const i32, sumk)
    };
    let b_ids = unsafe {
        std::slice::from_raw_parts(out_ids_b.contents() as *const i32, sumk)
    };
    let a_p = unsafe {
        std::slice::from_raw_parts(out_probs_a.contents() as *const f16, sumk)
    };
    let b_p = unsafe {
        std::slice::from_raw_parts(out_probs_b.contents() as *const f16, sumk)
    };
    assert_eq!(a_ids, b_ids);
    assert_eq!(a_p, b_p);
    // Invalid ids should be ignored; check lengths match counts
    let offsets = unsafe {
        std::slice::from_raw_parts(offsets_buf.contents() as *const u32, e + 1)
    };
    assert_eq!(offsets[e] as usize, sumk);
}
