use bytemuck::NoUninit;

/// M-RoPE layout of a batch: its rows are consecutive cells of a run of equal grids (a text token is a 1×1 grid).
///
/// - `first_cell`: cell of the batch's first row; row `i` is cell `first_cell + i`.
/// - `grid_width`: grid width in cells.
/// - `grid_height`: grid height in cells.
/// - `height_pairs`: number of pairs rotated by the cell's row.
/// - `width_pairs`: number of pairs rotated by the cell's column; the rest rotate by the grid's position.
/// - `merge_size`: cells within a grid go in raster order of `merge_size`×`merge_size` blocks.
/// - `interleaved_axes`: if nonzero, the axes alternate pair by pair (position, row, column, ...);
///   otherwise each axis takes a contiguous run of pairs.
#[derive(Clone, Copy, Debug, NoUninit)]
#[repr(C)]
pub struct RoPEGridParams {
    pub first_cell: u32,
    pub grid_width: u32,
    pub grid_height: u32,
    pub height_pairs: u32,
    pub width_pairs: u32,
    pub merge_size: u32,
    pub interleaved_axes: u32,
}
