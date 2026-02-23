use crate::DataType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Specialization {
    pub block_rows: i32,
    pub block_cols: i32,
    pub block_depth: i32,
    pub warps_per_row: u64,
    pub warps_per_col: u64,
    pub swizzle_log2: i32,
    pub align_m: bool,
    pub align_n: bool,
    pub align_k: bool,
}

impl Specialization {
    pub fn precompile_configs(_data_type: DataType) -> &'static [Self] {
        &[
            Self {
                block_rows: 64,
                block_cols: 64,
                block_depth: 256,
                warps_per_row: 2,
                warps_per_col: 2,
                swizzle_log2: 0,
                align_m: true,
                align_n: true,
                align_k: true,
            },
            Self {
                block_rows: 64,
                block_cols: 64,
                block_depth: 256,
                warps_per_row: 2,
                warps_per_col: 2,
                swizzle_log2: 0,
                align_m: false,
                align_n: true,
                align_k: true,
            },
            Self {
                block_rows: 64,
                block_cols: 64,
                block_depth: 256,
                warps_per_row: 2,
                warps_per_col: 2,
                swizzle_log2: 0,
                align_m: true,
                align_n: false,
                align_k: true,
            },
            Self {
                block_rows: 64,
                block_cols: 64,
                block_depth: 256,
                warps_per_row: 2,
                warps_per_col: 2,
                swizzle_log2: 0,
                align_m: false,
                align_n: false,
                align_k: true,
            },
            Self {
                block_rows: 128,
                block_cols: 128,
                block_depth: 512,
                warps_per_row: 4,
                warps_per_col: 4,
                swizzle_log2: 0,
                align_m: true,
                align_n: true,
                align_k: true,
            },
            Self {
                block_rows: 128,
                block_cols: 128,
                block_depth: 512,
                warps_per_row: 4,
                warps_per_col: 4,
                swizzle_log2: 0,
                align_m: false,
                align_n: true,
                align_k: true,
            },
        ]
    }
}
