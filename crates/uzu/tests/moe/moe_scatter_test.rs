#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::bf16;
use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::{
    common::kernel::MoeCountsOffsetsFusedKernel,
    metal::kernel::{
        KernelDataType, MoeBlockBasesArguments, MoeScatterArguments,
        MoeScatterKernels,
        dsl::MoeCountsOffsetsFusedMetalKernel,
        moe::{MoeRouterTopKArguments, MoeRouterTopKKernelWrapper},
    },
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

        // Generate random input and router weights for fused kernel
        let d_model = 64;
        let input_f32: Vec<f32> =
            (0..t * d_model).map(|_| rng.random_range(-1.0..1.0)).collect();
        let weight_f32: Vec<f32> =
            (0..e * d_model).map(|_| rng.random_range(-1.0..1.0)).collect();
        let bias_f32: Vec<f32> =
            (0..e).map(|_| rng.random_range(-0.5..0.5)).collect();

        let input: Vec<bf16> =
            input_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let weight: Vec<bf16> =
            weight_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let bias: Vec<bf16> =
            bias_f32.iter().map(|&x| bf16::from_f32(x)).collect();

        let input_buf = alloc_buffer_with_data(&ctx, &input);
        let weight_buf = alloc_buffer_with_data(&ctx, &weight);
        let bias_buf = alloc_buffer_with_data(&ctx, &bias);
        let topk_ids_buf = alloc_buffer::<i32>(&ctx, t * k);
        let topk_probs_buf = alloc_buffer::<bf16>(&ctx, t * k);

        // Use fused router+topk kernel
        let router_topk =
            MoeRouterTopKKernelWrapper::new(&ctx, KernelDataType::BFloat16)
                .expect("router_topk");
        let cb = ctx
            .command_queue
            .command_buffer()
            .expect("Failed to create command buffer");
        let _ = router_topk
            .encode(
                &cb,
                &MoeRouterTopKArguments {
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
                },
            )
            .expect("encode router_topk");
        cb.commit();
        cb.wait_until_completed();

        // Read back CPU probs for reference compare
        let probs_bf16 = unsafe {
            std::slice::from_raw_parts(
                topk_probs_buf.contents().as_ptr() as *const bf16,
                t * k,
            )
        };
        let topk_probs_cpu: Vec<f32> =
            probs_bf16.iter().map(|&h| f32::from(h)).collect();
        let topk_ids_cpu = unsafe {
            std::slice::from_raw_parts(
                topk_ids_buf.contents().as_ptr() as *const i32,
                t * k,
            )
        };

        let offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
        let sumk_buf = alloc_buffer::<u32>(&ctx, 1);
        let num_tiles = ((e + 511) / 512).max(1);
        let partials_buf = alloc_buffer::<u32>(&ctx, num_tiles * 512);

        let fused_kernel =
            MoeCountsOffsetsFusedMetalKernel::new(&ctx).expect("fused kernel");
        let cb = ctx
            .command_queue
            .command_buffer()
            .expect("Failed to create command buffer");
        let encoder = cb.new_compute_command_encoder().expect("encoder");
        fused_kernel.encode(
            &topk_ids_buf,
            &offsets_buf,
            &sumk_buf,
            &partials_buf,
            t as u32,
            e as u32,
            k as u32,
            &encoder,
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // Partials already created by fused kernel above
        let num_blocks = 1; // Fused kernel uses single block
        let num_tiles = ((e + 511) / 512).max(1);
        let entries = num_blocks * num_tiles * 512usize;
        let block_bases_buf = alloc_buffer::<u32>(&ctx, entries);

        let scatter = MoeScatterKernels::new(&ctx).expect("scatter kernels");
        let cb = ctx
            .command_queue
            .command_buffer()
            .expect("Failed to create command buffer");
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
            std::slice::from_raw_parts(
                sumk_buf.contents().as_ptr() as *const u32,
                1,
            )
        }[0] as usize;
        let out_ids_buf = alloc_buffer::<i32>(&ctx, sumk);
        let out_probs_buf = alloc_buffer::<bf16>(&ctx, sumk);

        let cb = ctx
            .command_queue
            .command_buffer()
            .expect("Failed to create command buffer");
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
                KernelDataType::BFloat16,
            )
            .expect("encode scatter");
        cb.commit();
        cb.wait_until_completed();

        let out_ids = unsafe {
            std::slice::from_raw_parts(
                out_ids_buf.contents().as_ptr() as *const i32,
                sumk,
            )
        };
        let out_probs_h = unsafe {
            std::slice::from_raw_parts(
                out_probs_buf.contents().as_ptr() as *const bf16,
                sumk,
            )
        };
        let _out_probs: Vec<f32> =
            out_probs_h.iter().map(|&h| f32::from(h)).collect();

        // CPU reference
        let (cpu_ids, _cpu_probs, offsets_cpu) =
            cpu_expert_buckets(topk_ids_cpu, &topk_probs_cpu, t, e, k);
        let offsets_gpu = unsafe {
            std::slice::from_raw_parts(
                offsets_buf.contents().as_ptr() as *const u32,
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
