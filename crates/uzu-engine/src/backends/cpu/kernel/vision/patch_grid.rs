/// Grid coordinates `(x, y)` of the `row`-th patch of the vision encoder input.
/// Same as `vision_patch_coordinates` in `metal/kernel/vision/patch_grid.h`, see the example there.
pub fn vision_patch_coordinates(
    row: u32,
    grid_width: u32,
    merge_size: u32,
) -> (u32, u32) {
    let block_size = merge_size * merge_size;
    let blocks_per_row = grid_width / merge_size;
    let block = row / block_size;
    let in_block = row % block_size;
    let y = (block / blocks_per_row) * merge_size + in_block / merge_size;
    let x = (block % blocks_per_row) * merge_size + in_block % merge_size;
    (x, y)
}
