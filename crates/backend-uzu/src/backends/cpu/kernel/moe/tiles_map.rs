use dsl::kernel;
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoeTileCounts)]
pub fn moe_tile_counts(
    #[allow(unused)] offsets: *const u32,
    #[allow(unused)] tile_counts: *mut u32,
    #[allow(unused)] e: u32,
) {
    todo!()
}

#[kernel(MoeTileScan)]
pub fn moe_tile_scan(
    #[allow(unused)] tile_counts: *const u32,
    #[allow(unused)] tile_row_offsets: *mut u32,
    #[allow(unused)] total_tiles_buf: *mut u32,
    #[allow(unused)] e: u32,
) {
    todo!()
}

#[kernel(MoeBuildTileMap)]
pub fn moe_build_tile_map(
    #[allow(unused)] offsets: *const u32,
    #[allow(unused)] tile_row_offsets: *const u32,
    #[allow(unused)] tile_counts: *const u32,
    #[allow(unused)] tile_map: *mut u32,
    #[allow(unused)] e: u32,
) {
    todo!()
}

#[kernel(MoeWriteDispatchArgs)]
pub fn moe_write_dispatch_args(
    #[allow(unused)] total_tiles_buf: *const u32,
    #[allow(unused)] dispatch_args: *mut u32,
    #[allow(unused)] num_tiles_n: u32,
) {
    todo!()
}
