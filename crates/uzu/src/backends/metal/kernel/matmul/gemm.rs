use crate::{
    DataType,
    backends::{
        common::{
            Context,
            kernel::matmul::{MatmulArgumentC, MatmulArguments},
        },
        metal::{Metal, context::MetalContext},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmSpecialization {
    pub block_rows: u32,
    pub block_cols: u32,
    pub block_depth: u32,
    pub simdgroups_per_row: u32,
    pub simdgroups_per_column: u32,
    pub align_mn: bool,
    pub align_k: bool,
    pub is_accumulate: bool,
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
                    align_mn: false,
                    align_k: true,
                    is_accumulate: false,
                },
                Self {
                    block_rows: 64,
                    block_cols: 32,
                    block_depth: 32,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    align_mn: true,
                    align_k: true,
                    is_accumulate: false,
                },
                Self {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    align_mn: false,
                    align_k: true,
                    is_accumulate: false,
                },
                Self {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    align_mn: true,
                    align_k: true,
                    is_accumulate: false,
                },
                Self {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 1,
                    simdgroups_per_column: 2,
                    align_mn: true,
                    align_k: true,
                    is_accumulate: false,
                },
            ],
            DataType::F16 => &[
                Self {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    align_mn: true,
                    align_k: true,
                    is_accumulate: false,
                },
                Self {
                    block_rows: 64,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    align_mn: false,
                    align_k: true,
                    is_accumulate: false,
                },
            ],
            DataType::F32 => &[
                Self {
                    block_rows: 32,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    align_mn: false,
                    align_k: true,
                    is_accumulate: false,
                },
                Self {
                    block_rows: 32,
                    block_cols: 64,
                    block_depth: 16,
                    simdgroups_per_row: 2,
                    simdgroups_per_column: 2,
                    align_mn: true,
                    align_k: true,
                    is_accumulate: false,
                },
            ],
            _ => &[],
        }
    }

    pub fn select(
        context: &MetalContext,
        data_type: DataType,
        arguments: &MatmulArguments<Metal>,
    ) -> Self {
        let overall_work_elements = arguments.batch_dim * arguments.output_dim;
        let is_float32 = matches!(data_type, DataType::F32);
        let prefer_half_or_tf32 = !is_float32 || MetalContext::tf32_enabled();

        let (block_rows, block_cols, block_depth, simdgroups_per_row, simdgroups_per_column) =
            if context.is_high_performance() && overall_work_elements >= (1 << 20) {
                if prefer_half_or_tf32 {
                    if 2 * std::cmp::max(arguments.batch_dim, arguments.output_dim) > arguments.input_dim {
                        (64, 64, 16, 2, 2)
                    } else {
                        (64, 32, 32, 2, 2)
                    }
                } else {
                    (32, 64, 16, 2, 2)
                }
            } else if prefer_half_or_tf32 {
                (64, 32, 32, 2, 2)
            } else {
                (32, 64, 16, 2, 2)
            };

        Self {
            block_rows,
            block_cols,
            block_depth,
            simdgroups_per_row,
            simdgroups_per_column,
            align_mn: (arguments.batch_dim % block_rows) == 0 && (arguments.output_dim % block_cols) == 0,
            align_k: (arguments.input_dim % block_depth) == 0,
            is_accumulate: matches!(arguments.c, MatmulArgumentC::Accumulate),
        }
    }
}
