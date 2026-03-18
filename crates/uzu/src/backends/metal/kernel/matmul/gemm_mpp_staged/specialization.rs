#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmMppStagedSpecialization {
    pub block_rows: i32,
    pub block_cols: i32,
    pub simdgroups_per_row: u64,
    pub simdgroups_per_column: u64,
    pub swizzle_log2: i32,
    pub align_m: bool,
    pub align_n: bool,
}

impl GemmMppStagedSpecialization {
    fn tile_config(
        block_rows: i32,
        block_cols: i32,
        simdgroups_per_row: u64,
        simdgroups_per_column: u64,
        align_m: bool,
        align_n: bool,
    ) -> Self {
        Self {
            block_rows,
            block_cols,
            simdgroups_per_row,
            simdgroups_per_column,
            swizzle_log2: 0,
            align_m,
            align_n,
        }
    }

    pub fn precompile_configs() -> Box<[Self]> {
        [
            Self::tile_config(64, 64, 2, 2, true, true),
            Self::tile_config(64, 64, 2, 2, false, true),
            Self::tile_config(64, 64, 2, 2, true, false),
            Self::tile_config(64, 64, 2, 2, false, false),
            Self::tile_config(32, 64, 2, 2, true, true),
            Self::tile_config(32, 64, 2, 2, false, true),
            Self::tile_config(64, 32, 4, 1, true, true),
            Self::tile_config(64, 32, 4, 1, true, false),
        ]
        .into()
    }

    pub fn select(
        m: i32,
        n: i32,
    ) -> Self {
        let (block_rows, block_cols, simdgroups_per_row, simdgroups_per_column) = if n < 64 {
            (64, 32, 4u64, 1u64)
        } else if m < 64 {
            (32, 64, 2u64, 2u64)
        } else {
            (64, 64, 2u64, 2u64)
        };

        Self {
            block_rows,
            block_cols,
            simdgroups_per_row,
            simdgroups_per_column,
            swizzle_log2: 0,
            align_m: (m % block_rows) == 0,
            align_n: (n % block_cols) == 0,
        }
    }
}
