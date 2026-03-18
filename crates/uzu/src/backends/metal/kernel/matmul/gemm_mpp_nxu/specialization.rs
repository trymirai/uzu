use crate::backends::metal::context::MetalContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmMppNxuSpecialization {
    pub block_rows: i32,
    pub block_cols: i32,
    pub simdgroups_per_row: u64,
    pub simdgroups_per_column: u64,
    pub swizzle_log2: i32,
    pub align_m: bool,
    pub align_n: bool,
    pub align_k: bool,
}

impl GemmMppNxuSpecialization {
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
            align_k: true,
        }
    }

    pub fn precompile_configs() -> Box<[Self]> {
        [
            Self::tile_config(64, 64, 2, 2, true, true),
            Self::tile_config(64, 64, 2, 2, false, true),
            Self::tile_config(64, 64, 2, 2, true, false),
            Self::tile_config(64, 64, 2, 2, false, false),
            Self::tile_config(128, 128, 4, 4, true, true),
            Self::tile_config(128, 128, 4, 4, false, true),
            Self::tile_config(128, 128, 4, 4, true, false),
            Self::tile_config(128, 128, 4, 4, false, false),
        ]
        .into()
    }

    pub fn select(
        context: &MetalContext,
        m: i32,
        n: i32,
        k: i32,
    ) -> Self {
        let _ = context;
        let large_problem = (m as i64) * (n as i64) >= (128 * 128);
        let (block_rows, block_cols, simdgroups_per_row, simdgroups_per_column) = if large_problem {
            (128, 128, 4u64, 4u64)
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
            align_k: (k % 256) == 0,
        }
    }
}
