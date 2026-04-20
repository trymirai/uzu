use dsl::kernel;

#[kernel(MoeTileCounts)]
pub fn moe_tile_counts(
    offsets: *const u32,
    tile_counts: *mut u32,
    e: u32,
) {
    let _ = (offsets, tile_counts, e);
    todo!()
}

#[kernel(MoeTileScan)]
pub fn moe_tile_scan(
    tile_counts: *const u32,
    tile_row_offsets: *mut u32,
    total_tiles_buf: *mut u32,
    e: u32,
) {
    let _ = (tile_counts, tile_row_offsets, total_tiles_buf, e);
    todo!()
}

#[kernel(MoeBuildTileMap)]
pub fn moe_build_tile_map(
    offsets: *const u32,
    tile_row_offsets: *const u32,
    tile_counts: *const u32,
    tile_map: *mut u32,
    e: u32,
) {
    let _ = (offsets, tile_row_offsets, tile_counts, tile_map, e);
    todo!()
}

#[kernel(MoeWriteDispatchArgs)]
pub fn moe_write_dispatch_args(
    total_tiles_buf: *const u32,
    dispatch_args: *mut u32,
    num_tiles_n: u32,
) {
    let _ = (total_tiles_buf, dispatch_args, num_tiles_n);
    todo!()
}
