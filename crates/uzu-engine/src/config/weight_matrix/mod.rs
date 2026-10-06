use uzu_engine_macros::{uzu_config, uzu_config_abstract};

pub mod d4s4_spec;
pub mod full_precision_spec;
pub mod hybrid_spec;
pub mod int_spec;
pub mod low_rank_spec;
pub mod mlx_spec;
pub mod qtip_gaussian;

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
    d4s4_spec::D4S4Spec,
    qtip_gaussian::QtipGaussianSpec,
)]
pub struct WeightMatrixSpec;
