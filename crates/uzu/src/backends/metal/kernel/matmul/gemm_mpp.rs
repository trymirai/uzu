#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmMppSpecialization {
    pub block_rows: u32,
    pub block_cols: u32,
    pub simdgroups_per_row: u32,
    pub simdgroups_per_column: u32,
    pub align_m: bool,
    pub align_n: bool,
}

impl GemmMppSpecialization {
    pub fn precompile_configs() -> &'static [Self] {
        &[
            Self {
                block_rows: 64,
                block_cols: 64,
                simdgroups_per_row: 2,
                simdgroups_per_column: 2,
                align_m: true,
                align_n: true,
            },
            Self {
                block_rows: 64,
                block_cols: 64,
                simdgroups_per_row: 2,
                simdgroups_per_column: 2,
                align_m: false,
                align_n: true,
            },
            Self {
                block_rows: 64,
                block_cols: 64,
                simdgroups_per_row: 2,
                simdgroups_per_column: 2,
                align_m: true,
                align_n: false,
            },
            Self {
                block_rows: 64,
                block_cols: 64,
                simdgroups_per_row: 2,
                simdgroups_per_column: 2,
                align_m: false,
                align_n: false,
            },
            Self {
                block_rows: 32,
                block_cols: 64,
                simdgroups_per_row: 2,
                simdgroups_per_column: 2,
                align_m: true,
                align_n: true,
            },
            Self {
                block_rows: 32,
                block_cols: 64,
                simdgroups_per_row: 2,
                simdgroups_per_column: 2,
                align_m: false,
                align_n: true,
            },
            Self {
                block_rows: 64,
                block_cols: 32,
                simdgroups_per_row: 4,
                simdgroups_per_column: 1,
                align_m: true,
                align_n: true,
            },
            Self {
                block_rows: 64,
                block_cols: 32,
                simdgroups_per_row: 4,
                simdgroups_per_column: 1,
                align_m: true,
                align_n: false,
            },
        ]
    }

    pub fn select(
        m: u32,
        n: u32,
    ) -> Self {
        let (block_rows, block_cols, simdgroups_per_row, simdgroups_per_column) = if n < 64 {
            (64, 32, 4u32, 1u32)
        } else if m < 64 {
            (32, 64, 2u32, 2u32)
        } else {
            (64, 64, 2u32, 2u32)
        };

        Self {
            block_rows,
            block_cols,
            simdgroups_per_row,
            simdgroups_per_column,
            align_m: (m % block_rows) == 0,
            align_n: (n % block_cols) == 0,
        }
    }
}
