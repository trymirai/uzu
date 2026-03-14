use crate::{
    DataType,
    backends::metal::{MetalDeviceCapabilities, context::MetalContext},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Specialization {
    pub block_rows: i32,
    pub block_cols: i32,
    pub block_depth: i32,
    pub warps_per_row: i32,
    pub warps_per_col: i32,
    pub swizzle_log2: i32,
    pub align_m: bool,
    pub align_n: bool,
    pub align_k: bool,
}

const LARGE_MATMUL_WORK_THRESHOLD: i64 = 1_048_576;

impl Specialization {
    fn tile_config(
        block_rows: i32,
        block_cols: i32,
        block_depth: i32,
        warps_per_row: i32,
        warps_per_col: i32,
        align_m: bool,
        align_n: bool,
    ) -> Self {
        Self {
            block_rows,
            block_cols,
            block_depth,
            warps_per_row,
            warps_per_col,
            swizzle_log2: 0,
            align_m,
            align_n,
            align_k: true,
        }
    }

    pub fn precompile_configs(capabilities: &MetalDeviceCapabilities) -> Box<[Self]> {
        let mut configs = vec![
            Self::tile_config(64, 32, 32, 2, 2, true, true),
            Self::tile_config(64, 32, 32, 2, 2, false, true),
            Self::tile_config(64, 32, 32, 2, 2, true, false),
            Self::tile_config(64, 32, 32, 2, 2, false, false),
        ];
        if capabilities.is_high_performance() {
            configs.extend([
                Self::tile_config(64, 64, 16, 2, 2, true, true),
                Self::tile_config(64, 64, 16, 2, 2, false, true),
                Self::tile_config(64, 64, 16, 2, 2, true, false),
                Self::tile_config(64, 64, 16, 2, 2, false, false),
            ]);
        }
        configs.into()
    }

    pub fn select(
        context: &MetalContext,
        data_type: DataType,
        m: i32,
        n: i32,
        k: i32,
    ) -> Self {
        let capabilities = context.device_capabilities();
        let work = (m as i64) * (n as i64);
        let reasonable_k = 2 * std::cmp::max(m, n) > k;

        let (block_rows, block_cols, block_depth, warps_per_row, warps_per_col) =
            if matches!(data_type, DataType::F32) {
                (32, 64, 16, 2, 2)
            } else if capabilities.is_high_performance()
                && work >= LARGE_MATMUL_WORK_THRESHOLD
                && reasonable_k
            {
                (64, 64, 16, 2, 2)
            } else {
                (64, 32, 32, 2, 2)
            };

        Self {
            block_rows,
            block_cols,
            block_depth,
            warps_per_row,
            warps_per_col,
            swizzle_log2: 0,
            align_m: (m % block_rows) == 0,
            align_n: (n % block_cols) == 0,
            align_k: (k % block_depth) == 0,
        }
    }
}
