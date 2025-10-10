#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::f16;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::kernel::{
    KernelDataType, MoeBlockBasesArguments, MoeCountsOffsetsFusedArguments,
    MoeCountsOffsetsFusedKernel, MoeScatterArguments, MoeScatterKernels,
    MoeTopKArguments, MoeTopKKernel,
};

use super::test_utils::{alloc_buffer, alloc_buffer_with_data, create_ctx};

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
    let ctx = create_ctx();
    let shapes = vec![(1usize, 4usize, 1usize), (7, 16, 2), (128, 64, 2)];
    for (t, e, k) in shapes {
        let mut rng = StdRng::seed_from_u64(7);
        let logits: Vec<f32> =
            (0..t * e).map(|_| rng.random_range(-4.0..4.0)).collect();
        let logits_buf = alloc_buffer_with_data(&ctx, &logits);
        let topk_ids_buf = alloc_buffer::<i32>(&ctx, t * k);
        let topk_probs_buf = alloc_buffer::<f16>(&ctx, t * k);
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

        let offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
        let sumk_buf = alloc_buffer::<u32>(&ctx, 1);
        let num_tiles = ((e + 511) / 512).max(1);
        let partials_buf = alloc_buffer::<u32>(&ctx, num_tiles * 512);

        let fused_kernel =
            MoeCountsOffsetsFusedKernel::new(&ctx).expect("fused kernel");
        let cb = ctx.command_queue.new_command_buffer();
        fused_kernel
            .encode(
                &cb,
                MoeCountsOffsetsFusedArguments {
                    topk_ids_buffer: &topk_ids_buf,
                    offsets_buffer: &offsets_buf,
                    sum_k_buffer: &sumk_buf,
                    partials_buffer: &partials_buf,
                    t,
                    e,
                    k,
                },
            )
            .expect("encode fused");
        cb.commit();
        cb.wait_until_completed();

        // Partials already created by fused kernel above
        let num_blocks = 1; // Fused kernel uses single block
        let num_tiles = ((e + 511) / 512).max(1);
        let entries = num_blocks * num_tiles * 512usize;
        let block_bases_buf = alloc_buffer::<u32>(&ctx, entries);

        let scatter = MoeScatterKernels::new(&ctx).expect("scatter kernels");
        let cb = ctx.command_queue.new_command_buffer();
        let block_alloc_buf = alloc_buffer::<u32>(&ctx, entries);
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
        let out_ids_buf = alloc_buffer::<i32>(&ctx, sumk);
        let out_probs_buf = alloc_buffer::<f16>(&ctx, sumk);

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
