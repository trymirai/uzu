use dsl::kernel;

#[kernel(MoePassATileCounts)]
pub fn moe_pass_a_tile_counts(
    expert_offsets: *const u32,
    tile_counts: *mut u32,
    e: u32,
    h_blocks: u32,
) {
    let _ = (expert_offsets, tile_counts, e, h_blocks);
    todo!()
}

#[kernel(MoePassATileScan)]
pub fn moe_pass_a_tile_scan(
    tile_counts: *const u32,
    tile_offsets: *mut u32,
    total_tiles: *mut u32,
    e: u32,
) {
    let _ = (tile_counts, tile_offsets, total_tiles, e);
    todo!()
}

#[kernel(MoePassABuildRowMap)]
pub fn moe_pass_a_build_row_map(
    expert_offsets: *const u32,
    row_expert_map: *mut u32,
    total_rows: u32,
    e: u32,
) {
    let _ = (expert_offsets, row_expert_map, total_rows, e);
    todo!()
}

#[kernel(MoePassABuildTileMap)]
pub fn moe_pass_a_build_tile_map(
    expert_offsets: *const u32,
    tile_offsets: *const u32,
    row_expert_map: *const u32,
    tile_map: *mut u32,
    total_rows: u32,
    h_blocks: u32,
) {
    let _ = (expert_offsets, tile_offsets, row_expert_map, tile_map, total_rows, h_blocks);
    todo!()
}

#[kernel(MoePassAWriteDispatchArgs)]
pub fn moe_pass_a_write_dispatch_args(
    total_tiles: *const u32,
    dispatch_args: *mut u32,
    num_tiles_y: u32,
) {
    let _ = (total_tiles, dispatch_args, num_tiles_y);
    todo!()
}
