use std::ops::{Deref, DerefMut};

use backend_uzu::{
    Array, ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{
            Backend, Encoder, Kernels,
            kernel::{
                MoeBlockBasesFromPartialsKernel, MoeCountsOffsetsFusedKernel, MoeRouterTopKKernel,
                MoeScatterBucketsKernel,
            },
        },
        cpu::Cpu,
    },
};
use half::bf16;
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{common::helpers::create_context, uzu_test};

fn cpu_expert_buckets<T: ArrayElement + Float>(
    topk_ids: &[i32],
    topk_probs: &[T],
    t: usize,
    e: usize,
    k: usize,
) -> (Vec<i32>, Vec<T>, Vec<u32>) {
    let mut per_e: Vec<Vec<(i32, T)>> = vec![Vec::new(); e];
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
    let mut probs = vec![T::zero(); sumk];
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

fn get_output_topk<B: Backend, T: ArrayElement + Float>(
    ctx: &B::Context,
    input: &[T],
    weights: &[T],
    bias: &[T],
    t: usize,
    d_model: usize,
    e: usize,
    k: usize,
) -> (Array<B>, Array<B>) {
    let input_array = ctx.create_array_from(&[input.len()], input, "");
    let weights_array = ctx.create_array_from(&[weights.len()], weights, "");
    let bias_array = ctx.create_array_from(&[bias.len()], bias, "");
    let topk_ids_array = ctx.create_array_uninitialized(&[t * k], DataType::I32, "");
    let topk_probs_array = ctx.create_array_uninitialized(&[t * k], T::data_type(), "");

    let topk_kernel =
        <<B as Backend>::Kernels as Kernels>::MoeRouterTopKKernel::new(&ctx, T::data_type()).expect("router_topk");
    let mut encoder = Encoder::new(ctx).expect("Failed to create encoder");
    topk_kernel.encode(
        input_array.buffer().borrow().deref(),
        weights_array.buffer().borrow().deref(),
        bias_array.buffer().borrow().deref(),
        topk_ids_array.buffer().borrow_mut().deref_mut(),
        topk_probs_array.buffer().borrow_mut().deref_mut(),
        t as u32,
        d_model as u32,
        e as u32,
        k as u32,
        true,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (topk_ids_array, topk_probs_array)
}

fn get_output<B: Backend, T: ArrayElement + Float>(
    input: &[T],
    weights: &[T],
    bias: &[T],
    t: usize,
    d_model: usize,
    e: usize,
    k: usize,
) -> (Vec<i32>, Vec<T>, Array<B>) {
    let ctx = create_context::<B>();

    // top-k
    let (topk_ids_array, topk_probs_array) =
        get_output_topk::<B, T>(ctx.as_ref(), input, weights, bias, t, d_model, e, k);

    // fused
    let offsets_array = ctx.create_array_uninitialized(&[e + 1], DataType::U32, "");
    let sumk_array = ctx.create_array_uninitialized(&[1], DataType::U32, "");
    let num_tiles = e.div_ceil(512).max(1);
    let partials_array = ctx.create_array_uninitialized(&[num_tiles * 512], DataType::U32, "");
    let fused_kernel =
        <<B as Backend>::Kernels as Kernels>::MoeCountsOffsetsFusedKernel::new(&ctx).expect("fused kernel");
    let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
    fused_kernel.encode(
        topk_ids_array.buffer().borrow().deref(),
        offsets_array.buffer().borrow_mut().deref_mut(),
        sumk_array.buffer().borrow_mut().deref_mut(),
        partials_array.buffer().borrow_mut().deref_mut(),
        t as u32,
        e as u32,
        k as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    // scatter bases
    // Partials already created by fused kernel above
    let num_blocks = 1; // Fused kernel uses single block
    let entries = num_blocks * num_tiles * 512usize;
    let block_bases_array = ctx.create_array_uninitialized(&[entries], DataType::U32, "");
    let block_alloc_array = ctx.create_array_uninitialized(&[entries], DataType::U32, "");
    let scatter_bases_kernel = <<B as Backend>::Kernels as Kernels>::MoeBlockBasesFromPartialsKernel::new(ctx.as_ref())
        .expect("Failed to create <<Metal as Backend>::Kernels as Kernels>::MoeBlockBasesFromPartialsKernel");
    let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
    scatter_bases_kernel.encode(
        partials_array.buffer().borrow().deref(),
        block_bases_array.buffer().borrow_mut().deref_mut(),
        block_alloc_array.buffer().borrow_mut().deref_mut(),
        e as u32,
        num_blocks as u32,
        num_tiles as u32,
        0u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let sumk = unsafe { std::slice::from_raw_parts(sumk_array.cpu_ptr().as_ptr() as *const u32, 1) }[0] as usize;

    // scatter
    let scatter_kernel = <<B as Backend>::Kernels as Kernels>::MoeScatterBucketsKernel::new(&ctx, T::data_type())
        .expect("Failed to create <<B as Backend>::Kernels as Kernels>::MoeScatterBucketsKernel");
    let out_ids_array = ctx.create_array_uninitialized(&[sumk], DataType::I32, "");
    let out_probs_array = ctx.create_array_uninitialized(&[sumk], T::data_type(), "");
    let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
    scatter_kernel.encode(
        topk_ids_array.buffer().borrow().deref(),
        topk_probs_array.buffer().borrow().deref(),
        offsets_array.buffer().borrow().deref(),
        block_bases_array.buffer().borrow().deref(),
        block_alloc_array.buffer().borrow().deref(),
        out_ids_array.buffer().borrow_mut().deref_mut(),
        out_probs_array.buffer().borrow_mut().deref_mut(),
        t as u32,
        e as u32,
        k as u32,
        num_blocks as u32,
        num_tiles as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (out_ids_array.as_slice().to_vec(), out_probs_array.as_slice().to_vec(), offsets_array)
}

fn test_scatter_internal<B: Backend, T: ArrayElement + Float>(
    t: usize,
    d_model: usize,
    e: usize,
    k: usize,
) {
    let mut rng = StdRng::seed_from_u64(7);

    // Generate random input and router weights for fused kernel
    let input: Vec<T> = (0..t * d_model).map(|_| T::from(rng.random_range(-1.0..1.0)).unwrap()).collect();
    let weight: Vec<T> = (0..e * d_model).map(|_| T::from(rng.random_range(-1.0..1.0)).unwrap()).collect();
    let bias: Vec<T> = (0..e).map(|_| T::from(rng.random_range(-0.5..0.5)).unwrap()).collect();

    let (out_ids, out_probs_h, out_offsets_array) =
        get_output::<B, T>(input.as_slice(), weight.as_slice(), bias.as_slice(), t, d_model, e, k);
    let _out_probs: Vec<f32> = out_probs_h.iter().map(|&h| h.to_f32().unwrap()).collect();

    // CPU reference
    let cpu_ctx = create_context::<Cpu>();
    let (topk_ids_cpu, topk_probs_cpu) = get_output_topk::<Cpu, T>(
        cpu_ctx.as_ref(),
        input.as_slice(),
        weight.as_slice(),
        bias.as_slice(),
        t,
        d_model,
        e,
        k,
    );
    let (cpu_ids, _cpu_probs, offsets_cpu) =
        cpu_expert_buckets(topk_ids_cpu.as_slice(), &topk_probs_cpu.as_slice::<T>(), t, e, k);
    let offsets_gpu = unsafe { std::slice::from_raw_parts(out_offsets_array.cpu_ptr().as_ptr() as *const u32, e + 1) };
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

#[uzu_test]
fn test_scatter_buckets_small() {
    for_each_non_cpu_backend!(|B| test_scatter_internal::<B, bf16>(1, 64, 4, 1));
}

#[uzu_test]
fn test_scatter_buckets_medium() {
    for_each_non_cpu_backend!(|B| test_scatter_internal::<B, bf16>(7, 64, 16, 2));
}

#[uzu_test]
fn test_scatter_buckets_big() {
    for_each_non_cpu_backend!(|B| test_scatter_internal::<B, bf16>(128, 64, 64, 2));
}
