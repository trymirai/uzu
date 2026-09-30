//! Fixed layout of the D4S4 embedding format (lalamo's `LatticeSpec` with kind `d4`).

pub const VALUES_PER_CODE: u32 = 4;
pub const CODEBOOK_SIZE: u32 = 256;
pub const COLUMNS_PER_LADDER_SCALE: u32 = 64;
pub const COLUMNS_PER_LADDER_INDEX_BYTE: u32 = 2 * COLUMNS_PER_LADDER_SCALE;
pub const LADDER_SIZE: u32 = 16;
