#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmMppDirectSpecialization {
    pub block_rows: u32,
    pub block_cols: u32,
    pub simdgroups_per_row: u32,
    pub simdgroups_per_column: u32,
    pub align_m: bool,
    pub align_n: bool,
    pub align_k: bool,
    pub bk: u32,
}

macro_rules! tile {
    ($br:expr, $bc:expr, $sr:expr, $sc:expr, $am:expr, $an:expr, $ak:expr, $bk:expr) => {
        GemmMppDirectSpecialization {
            block_rows: $br,
            block_cols: $bc,
            simdgroups_per_row: $sr,
            simdgroups_per_column: $sc,
            align_m: $am,
            align_n: $an,
            align_k: $ak,
            bk: $bk,
        }
    };
}

impl GemmMppDirectSpecialization {
    pub fn precompile_configs(data_type: crate::DataType) -> &'static [Self] {
        if !matches!(data_type, crate::DataType::F16 | crate::DataType::BF16) {
            return &[];
        }
        &[
            tile!(64, 64, 2, 2, true, true, true, 256),
            tile!(64, 64, 2, 2, true, true, false, 256),
            tile!(64, 64, 2, 2, false, true, false, 256),
            tile!(64, 64, 2, 2, true, false, false, 256),
            tile!(64, 64, 2, 2, false, false, false, 256),
            tile!(128, 128, 4, 4, true, true, true, 256),
            tile!(128, 128, 4, 4, true, true, false, 256),
            tile!(128, 128, 4, 4, false, true, false, 256),
            tile!(128, 128, 4, 4, true, false, false, 256),
            tile!(128, 128, 4, 4, false, false, false, 256),
        ]
    }

    pub fn select(m: u32, n: u32, k: u32) -> Self {
        let bk = 256u32;
        let large_problem = (m as u64) * (n as u64) >= (128 * 128);
        let (block_rows, block_cols, simdgroups_per_row, simdgroups_per_column) = if large_problem {
            (128, 128, 4u32, 4u32)
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
            align_k: (k % bk) == 0,
            bk,
        }
    }
}
