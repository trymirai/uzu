#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileConfiguration {
    pub tile_rows: i32,
    pub tile_cols: i32,
    pub tile_depth: i32,
    pub warps_per_row: i32,
    pub warps_per_col: i32,
}

pub fn select_tile_configuration(
    m: i32,
    n: i32,
) -> TileConfiguration {
    let tile_rows = if m < 40 {
        16
    } else {
        32
    };
    let tile_cols = if n < 40 {
        16
    } else {
        32
    };
    TileConfiguration {
        tile_rows,
        tile_cols,
        tile_depth: 16,
        warps_per_row: 2,
        warps_per_col: 2,
    }
}
