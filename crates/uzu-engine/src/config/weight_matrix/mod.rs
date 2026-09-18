use uzu_engine_macros::{uzu_config, uzu_config_abstract};

const GROUP_OUTPUT_ALIGNMENT: u32 = 4;

pub mod full_precision_spec;
pub mod hybrid_spec;
pub mod int_spec;
pub mod low_rank_spec;
pub mod mlx_spec;

#[uzu_config]
#[derive(Copy)]
#[serde(rename_all = "snake_case")]
pub enum QuantParamsLayout {
    OutputGroup,
    GroupOutput,
}

impl QuantParamsLayout {
    pub const fn row_stride(
        self,
        columns: u32,
        groups: u32,
    ) -> u32 {
        match self {
            Self::OutputGroup => groups,
            Self::GroupOutput => self.group_stride(columns),
        }
    }

    pub const fn group_stride(
        self,
        columns: u32,
    ) -> u32 {
        match self {
            Self::OutputGroup => 1,
            Self::GroupOutput => columns.next_multiple_of(GROUP_OUTPUT_ALIGNMENT),
        }
    }

    pub const fn plane_shape(
        self,
        columns: u32,
        groups: u32,
        packing_divisor: u32,
    ) -> [u32; 2] {
        let (rows, row_width) = match self {
            Self::OutputGroup => (columns, groups),
            Self::GroupOutput => (groups, self.group_stride(columns)),
        };
        [rows, row_width.div_ceil(packing_divisor)]
    }
}

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum WeightLayout {
    OutputInput,
    InputOutput,
}

#[uzu_config_abstract(
    full_precision_spec::FullPrecisionSpec,
    low_rank_spec::LowRankSpec,
    hybrid_spec::HybridSpec,
    int_spec::IntSpec,
    mlx_spec::MLXSpec
)]
pub struct WeightMatrixSpec;
