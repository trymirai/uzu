use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoeBlockBasesFromPartials)]
pub fn moe_block_bases_from_partials(
    partials: *const u32,
    block_bases: *mut u32,
    block_alloc: *mut u32,
    e_input: u32,
    num_blocks: u32,
    num_tiles: u32,
    capacity_per_expert: u32,
) {
    let _ = (partials, block_bases, block_alloc, e_input, num_blocks, num_tiles, capacity_per_expert);
    todo!()
}

#[kernel(MoeScatterBuckets)]
#[variants(T, f32, f16, bf16)]
pub fn moe_scatter_buckets<T: ArrayElement + Float>(
    topk_ids: *const i32,
    topk_probs: *const T,
    offsets: *const u32,
    block_bases: *const u32,
    block_alloc: *const u32,
    out_ids: *mut i32,
    out_probs: *mut T,
    t: u32,
    e: u32,
    k: u32,
    num_blocks: u32,
    num_tiles: u32,
) {
    let _ =
        (topk_ids, topk_probs, offsets, block_bases, block_alloc, out_ids, out_probs, t, e, k, num_blocks, num_tiles);
    todo!()
}

#[kernel(MoeScatterBucketsMap)]
#[variants(T, f32, f16, bf16)]
pub fn moe_scatter_buckets_map<T: ArrayElement + Float>(
    topk_ids: *const i32,
    topk_probs: *const T,
    offsets: *const u32,
    block_bases: *const u32,
    block_alloc: *const u32,
    out_ids: *mut i32,
    out_probs: *mut T,
    t: u32,
    e: u32,
    k: u32,
    num_blocks: u32,
    num_tiles: u32,
    tok2row: *mut i32,
) {
    let _ = (
        topk_ids,
        topk_probs,
        offsets,
        block_bases,
        block_alloc,
        out_ids,
        out_probs,
        t,
        e,
        k,
        num_blocks,
        num_tiles,
        tok2row,
    );
    todo!()
}
