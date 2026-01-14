#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileConfiguration {
    pub block_rows: i32,
    pub block_cols: i32,
    pub block_depth: i32,
}

impl TileConfiguration {
    pub fn new(
        block_rows: i32,
        block_cols: i32,
        block_depth: i32,
    ) -> Self {
        Self {
            block_rows,
            block_cols,
            block_depth,
        }
    }
}

pub fn select_tile_configuration(
    batch: i32,
    hidden_dim: i32,
) -> TileConfiguration {
    if batch >= 64 && hidden_dim >= 64 {
        TileConfiguration::new(64, 64, 16)
    } else {
        TileConfiguration::new(32, 32, 16)
    }
}
