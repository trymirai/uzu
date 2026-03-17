use crate::{
    DataType,
    backends::common::{Backend, Context, kernel::matmul::MatmulArguments},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmSpecialization {
    pub block_rows: i32,
    pub block_cols: i32,
    pub block_depth: i32,
    pub simdgroups_per_row: u64,
    pub simdgroups_per_column: u64,
    pub swizzle_log2: i32,
    pub align_m: bool,
    pub align_n: bool,
    pub align_k: bool,
}

impl GemmSpecialization {
    pub fn precompile_configs(data_type: DataType) -> &'static [Self] {
        match data_type {
            DataType::BF16 => &[
                Self {
                    block_rows: 64,
                    block_cols: 32,
                    block_depth: 32,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    swizzle_log2: 0,
                    align_m: false,
                    align_n: true,
                    align_k: true,
                },
                Self {
                    block_rows: 64,
                    block_cols: 32,
                    block_depth: 32,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    swizzle_log2: 0,
                    align_m: true,
                    align_n: true,
                    align_k: true,
                },
                Self {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    swizzle_log2: 0,
                    align_m: false,
                    align_n: true,
                    align_k: true,
                },
                Self {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    swizzle_log2: 0,
                    align_m: true,
                    align_n: false,
                    align_k: true,
                },
                Self {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    swizzle_log2: 0,
                    align_m: true,
                    align_n: true,
                    align_k: true,
                },
                Self {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 1,
                    simdgroups_per_column: 2,
                    swizzle_log2: 0,
                    align_m: true,
                    align_n: true,
                    align_k: true,
                },
            ],
            DataType::F16 => &[
                Self {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    swizzle_log2: 0,
                    align_m: true,
                    align_n: true,
                    align_k: true,
                },
                Self {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    swizzle_log2: 0,
                    align_m: false,
                    align_n: true,
                    align_k: true,
                },
            ],
            DataType::F32 => &[
                Self {
                    block_rows: 32,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    swizzle_log2: 0,
                    align_m: false,
                    align_n: true,
                    align_k: true,
                },
                Self {
                    block_rows: 32,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    swizzle_log2: 0,
                    align_m: true,
                    align_n: true,
                    align_k: true,
                },
            ],
            _ => &[],
        }
    }

    pub fn select<B: Backend>(
        context: &B::Context,
        data_type: DataType,
        arguments: &MatmulArguments<B>,
    ) -> Self {
        let overall_work_elements = (arguments.batch as i64) * (arguments.output_dim as i64);
        let is_float32 = matches!(data_type, DataType::F32);
        let prefer_half_or_tf32 = !is_float32 || B::Context::tf32_enabled();

        let (block_rows, block_cols, block_depth, simdgroups_per_row, simdgroups_per_column, swizzle_log2) =
            if context.is_high_performance() && overall_work_elements >= (1_i64 << 20) {
                if prefer_half_or_tf32 {
                    if 2 * std::cmp::max(arguments.batch, arguments.output_dim) > arguments.input_dim {
                        (64, 64, 16, 2, 2, 0)
                    } else if arguments.transpose_b {
                        (64, 32, 32, 2, 2, 0)
                    } else {
                        (32, 64, 16, 2, 2, 0)
                    }
                } else if arguments.transpose_b {
                    (32, 64, 16, 2, 2, 0)
                } else {
                    (64, 32, 32, 2, 2, 0)
                }
            } else if prefer_half_or_tf32 {
                if arguments.transpose_b {
                    (64, 32, 32, 2, 2, 0)
                } else {
                    (64, 64, 16, 2, 2, 0)
                }
            } else if arguments.transpose_b {
                (32, 64, 16, 2, 2, 0)
            } else {
                (64, 32, 32, 2, 2, 0)
            };

        let m = arguments.batch;
        let n = arguments.output_dim;
        let k = arguments.input_dim;

        Self {
            block_rows,
            block_cols,
            block_depth,
            simdgroups_per_row,
            simdgroups_per_column,
            swizzle_log2,
            align_m: (m % block_rows) == 0,
            align_n: (n % block_cols) == 0,
            align_k: (k % block_depth) == 0,
        }
    }
}
