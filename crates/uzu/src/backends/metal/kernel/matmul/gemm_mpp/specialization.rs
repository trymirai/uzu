use crate::backends::metal::MetalDeviceCapabilities;

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
    pub use_native_fragment_layout: bool,
    pub subtile_rows: u32,
    pub subtile_cols: u32,
    pub matmul_k_step: u32,
}

impl Specialization {
    pub const SUBTILE_ROWS: u32 = 16;
    pub const SUBTILE_COLS: u32 = 32;
    pub const MATMUL_K_STEP: u32 = 16;

    fn tile_config(
        block_rows: i32,
        block_cols: i32,
        warps_per_row: u64,
        warps_per_col: u64,
        align_m: bool,
        align_n: bool,
        threadgroup_memory: i32,
    ) -> Self {
        Self {
            block_rows,
            block_cols,
            block_depth: threadgroup_memory,
            warps_per_row,
            warps_per_col,
            swizzle_log2: 0,
            align_m,
            align_n,
            align_k: true,
            use_native_fragment_layout: false,
            subtile_rows: Self::SUBTILE_ROWS,
            subtile_cols: Self::SUBTILE_COLS,
            matmul_k_step: Self::MATMUL_K_STEP,
        }
    }

    pub fn precompile_configs(capabilities: &MetalDeviceCapabilities) -> Box<[Self]> {
        let threadgroup_memory = capabilities.max_threadgroup_memory.as_u64() as i32;
        [
            Self::tile_config(64, 64, 2, 2, true, true, threadgroup_memory),
            Self::tile_config(64, 64, 2, 2, false, true, threadgroup_memory),
            Self::tile_config(64, 64, 2, 2, true, false, threadgroup_memory),
            Self::tile_config(64, 64, 2, 2, false, false, threadgroup_memory),
            Self::tile_config(32, 64, 2, 2, true, true, threadgroup_memory),
            Self::tile_config(32, 64, 2, 2, false, true, threadgroup_memory),
            Self::tile_config(64, 32, 4, 1, true, true, threadgroup_memory),
            Self::tile_config(64, 32, 4, 1, true, false, threadgroup_memory),
        ]
        .into()
    }

    pub fn select(
        capabilities: &MetalDeviceCapabilities,
        m: i32,
        n: i32,
    ) -> Self {
        let threadgroup_memory_budget = capabilities.max_threadgroup_memory.as_u64() as i32;

        let (block_rows, block_cols, warps_per_row, warps_per_col) = if n < 64 {
            (64, 32, 4u64, 1u64)
        } else if m < 64 {
            (32, 64, 2u64, 2u64)
        } else {
            (64, 64, 2u64, 2u64)
        };

        Self {
            block_rows,
            block_cols,
            block_depth: threadgroup_memory_budget,
            warps_per_row,
            warps_per_col,
            swizzle_log2: 0,
            align_m: (m % block_rows) == 0,
            align_n: (n % block_cols) == 0,
            align_k: true,
            use_native_fragment_layout: capabilities.supports_mxu,
            subtile_rows: Self::SUBTILE_ROWS,
            subtile_cols: Self::SUBTILE_COLS,
            matmul_k_step: Self::MATMUL_K_STEP,
        }
    }
}
