use bytemuck::NoUninit;

/// How AttentionPrepare rotates queries and keys.
///
/// - `None`: no rotation.
/// - `Sequential`: every pair rotates by the row's position.
/// - `ContiguousGrid`: rows are cells of `RoPEGridParams` grids; each axis (position, row, column) takes a contiguous
///   run of pairs.
/// - `InterleavedGrid`: like `ContiguousGrid`, but the axes alternate pair by pair (position, row, column, ...).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoPEKind {
    None = 0,
    Sequential = 1,
    ContiguousGrid = 2,
    InterleavedGrid = 3,
}

/// M-RoPE layout of a batch: its rows are consecutive cells of a run of equal grids (a text token is a 1×1 grid).
///
/// - `first_cell`: cell of the batch's first row; row `i` is cell `first_cell + i`.
/// - `grid_width`: grid width in cells.
/// - `grid_height`: grid height in cells.
/// - `height_pairs`: number of pairs rotated by the cell's row.
/// - `width_pairs`: number of pairs rotated by the cell's column; the rest rotate by the grid's position.
/// - `merge_size`: cells within a grid go in raster order of `merge_size`×`merge_size` blocks.
#[derive(Clone, Copy, Debug, NoUninit)]
#[repr(C)]
pub struct RoPEGridParams {
    pub first_cell: u32,
    pub grid_width: u32,
    pub grid_height: u32,
    pub height_pairs: u32,
    pub width_pairs: u32,
    pub merge_size: u32,
}
