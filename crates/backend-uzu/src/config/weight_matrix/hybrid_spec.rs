use proc_macros::uzu_config;

use crate::config::weight_matrix::AnyWeightMatrixSpec;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum IncoherenceProcessingMode {
    Input,
    Output,
    InputOutput,
}

#[uzu_config(super::WeightMatrixSpec)]
pub struct HybridSpec {
    pub quantization_spec: Box<AnyWeightMatrixSpec>,
    pub adapter_spec: Option<Box<AnyWeightMatrixSpec>>,
    pub incoherence_block_size: Option<usize>,
    pub incoherence_processing_mode: IncoherenceProcessingMode,
}
