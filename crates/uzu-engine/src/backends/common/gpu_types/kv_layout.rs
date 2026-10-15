pub const TILE_KEYS: u32 = 32;
pub const TILE_DIMS: u32 = 16;
pub const FRAGMENT_BYTES: u32 = TILE_KEYS * TILE_DIMS;
pub const SLICE_DIMS: u32 = 64;
pub const V_CODE_BIAS: i16 = 128;

pub const GROUP_ROWS: u32 = 8;
pub const LANE_COLS: u32 = 4;
pub const LANE_BYTES: u32 = TILE_KEYS / GROUP_ROWS * LANE_COLS;

pub const WORD_BITS: u32 = 32;
pub const ROTATION_SIGN_BITS: [u32; 8] =
    [0x9e3779b9, 0x85ebca6b, 0xc2b2ae35, 0x27d4eb2f, 0x165667b1, 0xbf58476d, 0x94d049bb, 0xd6e8feb8];
