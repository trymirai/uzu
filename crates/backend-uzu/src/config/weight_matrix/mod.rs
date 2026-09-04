use proc_macros::{uzu_config, uzu_config_abstract};

pub mod d4_s4_spec;
pub mod full_precision_spec;
pub mod hybrid_spec;
pub mod i3_s4_spec;
pub mod int_spec;
pub mod low_rank_spec;
pub mod mlx_spec;
pub mod qtip_gaussian_spec;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    OutputInput,
    InputOutput,
}

#[uzu_config_abstract(
    full_precision_spec::FullPrecisionSpec,
    low_rank_spec::LowRankSpec,
    hybrid_spec::HybridSpec,
    int_spec::IntSpec,
    mlx_spec::MLXSpec,
    qtip_gaussian_spec::QtipGaussianSpec,
    d4_s4_spec::D4S4Spec,
    i3_s4_spec::I3S4Spec
)]
pub struct WeightMatrixSpec;
