use uzu_engine_macros::{uzu_config, uzu_config_abstract};

pub mod d4s4_spec;
pub mod full_precision_spec;
pub mod hybrid_spec;
pub mod int_spec;
pub mod mlx_spec;
pub mod output_stack;
pub mod qtip_gaussian;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    OutputInput,
    InputOutput,
}

#[uzu_config_abstract(
    full_precision_spec::FullPrecisionSpec,
    hybrid_spec::HybridSpec,
    int_spec::IntSpec,
    mlx_spec::MLXSpec,
    d4s4_spec::D4S4Spec,
    d4s4_spec::LatticeSpec,
    qtip_gaussian::QtipGaussianSpec,
    output_stack::OutputStackSpec
)]
pub struct WeightMatrixSpec;
