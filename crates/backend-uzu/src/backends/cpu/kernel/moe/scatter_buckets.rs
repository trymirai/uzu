use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoeBlockBasesFromPartials)]
pub fn moe_block_bases_from_partials(
    #[allow(unused)] partials: *const u32,
    #[allow(unused)] block_bases: *mut u32,
    #[allow(unused)] block_alloc: *mut u32,
    #[allow(unused)] e_input: u32,
    #[allow(unused)] num_blocks: u32,
    #[allow(unused)] num_tiles: u32,
    #[allow(unused)] capacity_per_expert: u32,
) {
    todo!()
}

#[kernel(MoeScatterBuckets)]
#[variants(T, f32, f16, bf16)]
pub fn moe_scatter_buckets<T: ArrayElement + Float>(
    #[allow(unused)] topk_ids: *const i32,
    #[allow(unused)] topk_probs: *const T,
    #[allow(unused)] offsets: *const u32,
    #[allow(unused)] block_bases: *const u32,
    #[allow(unused)] block_alloc: *const u32,
    #[allow(unused)] out_ids: *mut i32,
    #[allow(unused)] out_probs: *mut T,
    #[allow(unused)] t: u32,
    #[allow(unused)] e: u32,
    #[allow(unused)] k: u32,
    #[allow(unused)] num_blocks: u32,
    #[allow(unused)] num_tiles: u32,
) {
    todo!()
}

#[kernel(MoeScatterBucketsMap)]
#[variants(T, f32, f16, bf16)]
pub fn moe_scatter_buckets_map<T: ArrayElement + Float>(
    #[allow(unused)] topk_ids: *const i32,
    #[allow(unused)] topk_probs: *const T,
    #[allow(unused)] offsets: *const u32,
    #[allow(unused)] block_bases: *const u32,
    #[allow(unused)] block_alloc: *const u32,
    #[allow(unused)] out_ids: *mut i32,
    #[allow(unused)] out_probs: *mut T,
    #[allow(unused)] t: u32,
    #[allow(unused)] e: u32,
    #[allow(unused)] k: u32,
    #[allow(unused)] num_blocks: u32,
    #[allow(unused)] num_tiles: u32,
    #[allow(unused)] tok2row: *mut i32,
) {
    todo!()
}
