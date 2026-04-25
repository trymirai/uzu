use dsl::kernel;
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoePassATileCounts)]
pub fn moe_pass_a_tile_counts(
    #[allow(unused)] expert_offsets: *const u32,
    #[allow(unused)] tile_counts: *mut u32,
    #[allow(unused)] e: u32,
    #[allow(unused)] h_blocks: u32,
) {
    todo!()
}

#[kernel(MoePassATileScan)]
pub fn moe_pass_a_tile_scan(
    #[allow(unused)] tile_counts: *const u32,
    #[allow(unused)] tile_offsets: *mut u32,
    #[allow(unused)] total_tiles: *mut u32,
    #[allow(unused)] e: u32,
) {
    todo!()
}

#[kernel(MoePassABuildRowMap)]
pub fn moe_pass_a_build_row_map(
    #[allow(unused)] expert_offsets: *const u32,
    #[allow(unused)] row_expert_map: *mut u32,
    #[allow(unused)] total_rows: u32,
    #[allow(unused)] e: u32,
) {
    todo!()
}

#[kernel(MoePassABuildTileMap)]
pub fn moe_pass_a_build_tile_map(
    #[allow(unused)] expert_offsets: *const u32,
    #[allow(unused)] tile_offsets: *const u32,
    #[allow(unused)] row_expert_map: *const u32,
    #[allow(unused)] tile_map: *mut u32,
    #[allow(unused)] total_rows: u32,
    #[allow(unused)] h_blocks: u32,
) {
    todo!()
}

#[kernel(MoePassAWriteDispatchArgs)]
pub fn moe_pass_a_write_dispatch_args(
    #[allow(unused)] total_tiles: *const u32,
    #[allow(unused)] dispatch_args: *mut u32,
    #[allow(unused)] num_tiles_y: u32,
) {
    todo!()
}
