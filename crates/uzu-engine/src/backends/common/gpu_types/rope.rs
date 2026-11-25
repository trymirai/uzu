use bytemuck::NoUninit;

/// M-RoPE layout of a batch: its rows are cells `first_cell..` of a run of `grid_width`×`grid_height` grids (a text
/// token is a 1×1 grid). Cells go in raster order of `merge_size`×`merge_size` blocks. `height_pairs` pairs rotate
/// by the cell's row, `width_pairs` by its column, the rest by the grid's position. With `interleaved_axes` the axes
/// alternate pair by pair (position, row, column, ...), otherwise each axis takes a contiguous run of pairs.
#[derive(Clone, Copy, Debug, NoUninit)]
#[repr(C)]
pub struct RoPEGridParams {
    pub grid_width: u32,
    pub grid_height: u32,
    pub first_cell: u32,
    pub height_pairs: u32,
    pub width_pairs: u32,
    pub merge_size: u32,
    pub interleaved_axes: u32,
}
