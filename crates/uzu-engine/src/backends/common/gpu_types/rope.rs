use bytemuck::NoUninit;

/// M-RoPE layout of a batch: its rows are cells `first_cell..` of a run of `grid_width`×`grid_height` grids in raster
/// order (a text token is a 1×1 grid). Pairs `3i + 1` below `3 * height_pairs` rotate by the cell's row, pairs
/// `3i + 2` below `3 * width_pairs` by its column, the rest by the grid's position.
#[derive(Clone, Copy, Debug, NoUninit)]
#[repr(C)]
pub struct RoPEGridParams {
    pub grid_width: u32,
    pub grid_height: u32,
    pub first_cell: u32,
    pub height_pairs: u32,
    pub width_pairs: u32,
}
