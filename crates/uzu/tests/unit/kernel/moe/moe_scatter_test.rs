#![cfg(metal_backend)]

use half::bf16;
use metal::MTLBuffer;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::{
    DataType,
    backends::{
        common::{
            Backend, Encoder, Kernels,
            kernel::{
                MoeBlockBasesFromPartialsKernel, MoeCountsOffsetsFusedKernel, MoeRouterTopKKernel,
                MoeScatterBucketsKernel,
            },
        },
        metal::Metal,
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
        let input_f32: Vec<f32> = (0..t * d_model).map(|_| rng.random_range(-1.0..1.0)).collect();
        let weight_f32: Vec<f32> = (0..e * d_model).map(|_| rng.random_range(-1.0..1.0)).collect();
        let bias_f32: Vec<f32> = (0..e).map(|_| rng.random_range(-0.5..0.5)).collect();

        let input: Vec<bf16> = input_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let weight: Vec<bf16> = weight_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let bias: Vec<bf16> = bias_f32.iter().map(|&x| bf16::from_f32(x)).collect();

        let input_buf = alloc_buffer_with_data(&ctx, &input);
        let weight_buf = alloc_buffer_with_data(&ctx, &weight);
        let bias_buf = alloc_buffer_with_data(&ctx, &bias);
        let mut topk_ids_buf = alloc_buffer::<i32>(&ctx, t * k);
        let mut topk_probs_buf = alloc_buffer::<bf16>(&ctx, t * k);

        // Use fused router+topk kernel
        let router_topk = <<Metal as Backend>::Kernels as Kernels>::MoeRouterTopKKernel::new(&ctx, DataType::BF16)
            .expect("router_topk");
        let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
        router_topk.encode(
            &input_buf,
            &weight_buf,
            &bias_buf,
            &mut topk_ids_buf,
            &mut topk_probs_buf,
            t as u32,
            d_model as u32,
            e as u32,
            k as u32,
            true,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();

        // Read back CPU probs for reference compare
        let probs_bf16 =
            unsafe { std::slice::from_raw_parts(topk_probs_buf.contents().as_ptr() as *const bf16, t * k) };
        let topk_probs_cpu: Vec<f32> = probs_bf16.iter().map(|&h| f32::from(h)).collect();
        let topk_ids_cpu = unsafe { std::slice::from_raw_parts(topk_ids_buf.contents().as_ptr() as *const i32, t * k) };

        let mut offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
        let mut sumk_buf = alloc_buffer::<u32>(&ctx, 1);
        let num_tiles = ((e + 511) / 512).max(1);
        let mut partials_buf = alloc_buffer::<u32>(&ctx, num_tiles * 512);

        let fused_kernel =
            <<Metal as Backend>::Kernels as Kernels>::MoeCountsOffsetsFusedKernel::new(&ctx).expect("fused kernel");
        let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
        fused_kernel.encode(
            &topk_ids_buf,
            &mut offsets_buf,
            &mut sumk_buf,
            &mut partials_buf,
            t as u32,
            e as u32,
            k as u32,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();

        // Partials already created by fused kernel above
        let num_blocks = 1; // Fused kernel uses single block
        let num_tiles = ((e + 511) / 512).max(1);
        let entries = num_blocks * num_tiles * 512usize;
        let mut block_bases_buf = alloc_buffer::<u32>(&ctx, entries);

        let scatter_bases_kernel = <<Metal as Backend>::Kernels as Kernels>::MoeBlockBasesFromPartialsKernel::new(&ctx)
            .expect("Failed to create <<Metal as Backend>::Kernels as Kernels>::MoeBlockBasesFromPartialsKernel");
        let scatter_kernel =
            <<Metal as Backend>::Kernels as Kernels>::MoeScatterBucketsKernel::new(&ctx, DataType::BF16)
                .expect("Failed to create <<Metal as Backend>::Kernels as Kernels>::MoeScatterBucketsKernel");
        let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
        let mut block_alloc_buf = alloc_buffer::<u32>(&ctx, entries);

        scatter_bases_kernel.encode(
            &partials_buf,
            &mut block_bases_buf,
            &mut block_alloc_buf,
            e as u32,
            num_blocks as u32,
            num_tiles as u32,
            0u32,
            &mut encoder,
        );

        encoder.end_encoding().submit().wait_until_completed().unwrap();

        let sumk = unsafe { std::slice::from_raw_parts(sumk_buf.contents().as_ptr() as *const u32, 1) }[0] as usize;
        let mut out_ids_buf = alloc_buffer::<i32>(&ctx, sumk);
        let mut out_probs_buf = alloc_buffer::<bf16>(&ctx, sumk);

        let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
        scatter_kernel.encode(
            &topk_ids_buf,
            &topk_probs_buf,
            &offsets_buf,
            &block_bases_buf,
            &block_alloc_buf,
            &mut out_ids_buf,
            &mut out_probs_buf,
            t as u32,
            e as u32,
            k as u32,
            num_blocks as u32,
            num_tiles as u32,
            &mut encoder,
        );

        encoder.end_encoding().submit().wait_until_completed().unwrap();

        let out_ids = unsafe { std::slice::from_raw_parts(out_ids_buf.contents().as_ptr() as *const i32, sumk) };
        let out_probs_h = unsafe { std::slice::from_raw_parts(out_probs_buf.contents().as_ptr() as *const bf16, sumk) };
        let _out_probs: Vec<f32> = out_probs_h.iter().map(|&h| f32::from(h)).collect();

        // CPU reference
        let (cpu_ids, _cpu_probs, offsets_cpu) = cpu_expert_buckets(topk_ids_cpu, &topk_probs_cpu, t, e, k);
        let offsets_gpu = unsafe { std::slice::from_raw_parts(offsets_buf.contents().as_ptr() as *const u32, e + 1) };
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
