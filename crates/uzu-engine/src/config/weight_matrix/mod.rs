use uzu_engine_macros::{uzu_config, uzu_config_abstract};

pub mod d4s4_spec;
pub mod full_precision_spec;
mod hadamard_spec;
pub mod hybrid_spec;
mod hybrid_spec_input;
mod incoherence_processing_mode;
pub mod int_spec;
pub mod mlx_spec;

pub use hadamard_spec::HadamardSpec;
pub use hybrid_spec::HybridSpec;
pub use hybrid_spec_input::HybridSpecInput;
pub use incoherence_processing_mode::IncoherenceProcessingMode;

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
    d4s4_spec::D4S4Spec
)]
pub struct WeightMatrixSpec;
