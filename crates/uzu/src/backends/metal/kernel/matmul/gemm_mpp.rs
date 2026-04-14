use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmMppSpecialization {
    pub block_rows: u32,
    pub block_cols: u32,
    pub simdgroups_per_row: u32,
    pub simdgroups_per_column: u32,
    pub align_m: bool,
    pub align_n: bool,
    pub align_k: bool,
    pub apply_ab_scale: bool,
    pub is_accumulate: bool,
}

const fn config(
    block_rows: u32,
    block_cols: u32,
    simdgroups_per_row: u32,
    simdgroups_per_column: u32,
    align_m: bool,
    align_n: bool,
    is_accumulate: bool,
) -> GemmMppSpecialization {
    GemmMppSpecialization {
        block_rows,
        block_cols,
        simdgroups_per_row,
        simdgroups_per_column,
        align_m,
        align_n,
        align_k: false,
        apply_ab_scale: false,
        is_accumulate,
    }
}

static GEMM_MPP_PRECOMPILE: OnceLock<&'static [GemmMppSpecialization]> = OnceLock::new();

impl GemmMppSpecialization {
    pub fn precompile_configs(data_type: crate::DataType) -> &'static [Self] {
        if !matches!(data_type, crate::DataType::F16 | crate::DataType::BF16) {
            return &[];
        }
        GEMM_MPP_PRECOMPILE.get_or_init(|| {
            const BASE: &[GemmMppSpecialization] = &[
                config(64, 64, 2, 2, true, true, false),
                config(64, 64, 2, 2, false, true, false),
                config(64, 64, 2, 2, true, false, false),
                config(64, 64, 2, 2, false, false, false),
                config(32, 64, 2, 2, true, true, false),
                config(32, 64, 2, 2, false, true, false),
                config(64, 32, 4, 1, true, true, false),
                config(64, 32, 4, 1, true, false, false),
                config(128, 128, 4, 4, true, true, false),
                config(128, 128, 4, 4, false, true, false),
                config(128, 128, 4, 4, true, false, false),
                config(128, 128, 4, 4, false, false, false),
                config(64, 64, 2, 2, true, true, true),
                config(64, 64, 2, 2, false, true, true),
                config(64, 64, 2, 2, true, false, true),
                config(64, 64, 2, 2, false, false, true),
                config(32, 64, 2, 2, true, true, true),
                config(32, 64, 2, 2, false, true, true),
                config(64, 32, 4, 1, true, true, true),
                config(64, 32, 4, 1, true, false, true),
                config(128, 128, 4, 4, true, true, true),
                config(128, 128, 4, 4, false, true, true),
                config(128, 128, 4, 4, true, false, true),
                config(128, 128, 4, 4, false, false, true),
            ];

            let mut expanded = Vec::with_capacity(BASE.len() * 4);
            for &base in BASE {
                expanded.push(Self {
                    align_k: true,
                    apply_ab_scale: true,
                    ..base
                });
                expanded.push(Self {
                    align_k: true,
                    apply_ab_scale: false,
                    ..base
                });
                expanded.push(Self {
                    align_k: false,
                    apply_ab_scale: true,
                    ..base
                });
                expanded.push(Self {
                    align_k: false,
                    apply_ab_scale: false,
                    ..base
                });
            }

            Box::leak(expanded.into_boxed_slice())
        })
    }

    pub fn select(
        m: u32,
        n: u32,
        k: u32,
        is_accumulate: bool,
        apply_ab_scale: bool,
    ) -> Self {
        let (block_rows, block_cols, simdgroups_per_row, simdgroups_per_column) = if m >= 128 && n >= 128 {
            (128, 128, 4u32, 4u32)
        } else if n < 64 {
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
            align_k: (k % 256) == 0,
            apply_ab_scale,
            is_accumulate,
        }
    }
}
