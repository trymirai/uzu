#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVLayout {
    RowMajor = 0,
    MxuTiled = 1,
}

pub const TILE_KEYS: u32 = 32;
pub const TILE_QUERIES: u32 = 16;
pub const TILE_DIMS: u32 = 16;
pub const FRAGMENT_BYTES: u32 = TILE_KEYS * TILE_DIMS;
pub const SLICE_DIMS: u32 = 64;
pub const GROUP_ROWS: u32 = 8;
pub const LANE_COLS: u32 = 4;
pub const LANE_BYTES: u32 = TILE_KEYS / GROUP_ROWS * LANE_COLS;
