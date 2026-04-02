#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmMppSpecialization {
    pub block_rows: i32,
    pub block_cols: i32,
    pub simdgroups_per_row: u64,
    pub simdgroups_per_column: u64,
    pub swizzle_log2: i32,
    pub align_m: bool,
    pub align_n: bool,
}

impl GemmMppSpecialization {
    pub fn precompile_configs() -> Box<[Self]> {
        [
            Self {
                block_rows: 64,
                block_cols: 64,
                simdgroups_per_row: 2,
                simdgroups_per_column: 2,
                swizzle_log2: 0,
                align_m: true,
                align_n: true,
            },
            Self {
                block_rows: 64,
                block_cols: 64,
                simdgroups_per_row: 2,
                simdgroups_per_column: 2,
                swizzle_log2: 0,
                align_m: false,
                align_n: true,
            },
            Self {
                block_rows: 64,
                block_cols: 64,
                simdgroups_per_row: 2,
                simdgroups_per_column: 2,
                swizzle_log2: 0,
                align_m: true,
                align_n: false,
            },
            Self {
                block_rows: 64,
                block_cols: 64,
                simdgroups_per_row: 2,
                simdgroups_per_column: 2,
                swizzle_log2: 0,
                align_m: false,
                align_n: false,
            },
            Self {
                block_rows: 32,
                block_cols: 64,
                simdgroups_per_row: 2,
                simdgroups_per_column: 2,
                swizzle_log2: 0,
                align_m: true,
                align_n: true,
            },
            Self {
                block_rows: 32,
                block_cols: 64,
                simdgroups_per_row: 2,
                simdgroups_per_column: 2,
                swizzle_log2: 0,
                align_m: false,
                align_n: true,
            },
            Self {
                block_rows: 64,
                block_cols: 32,
                simdgroups_per_row: 4,
                simdgroups_per_column: 1,
                swizzle_log2: 0,
                align_m: true,
                align_n: true,
            },
            Self {
                block_rows: 64,
                block_cols: 32,
                simdgroups_per_row: 4,
                simdgroups_per_column: 1,
                swizzle_log2: 0,
                align_m: true,
                align_n: false,
            },
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
