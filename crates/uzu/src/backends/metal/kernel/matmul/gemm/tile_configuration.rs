#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileConfiguration {
    pub block_rows: i32,
    pub block_cols: i32,
    pub block_depth: i32,
    pub warps_per_row: u64,
    pub warps_per_col: u64,
    pub swizzle_log2: i32,
}

impl TileConfiguration {
    pub fn new(
        block_rows: i32,
        block_cols: i32,
        block_depth: i32,
        warps_per_row: u64,
        warps_per_col: u64,
        swizzle_log2: i32,
    ) -> Self {
        Self {
            block_rows,
            block_cols,
            block_depth,
            warps_per_row,
            warps_per_col,
            swizzle_log2,
        }
    }

    pub fn is_nax(&self) -> bool {
        self.block_depth >= 256
    }
}
