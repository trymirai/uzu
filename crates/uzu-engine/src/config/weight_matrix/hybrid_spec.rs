use uzu_engine_macros::{uzu_config, uzu_config_abstract};

use crate::{config::weight_matrix::AnyWeightMatrixSpec, utils::strict_serde::Unsupported};

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum IncoherenceProcessingMode {
    Input,
    Output,
    InputOutput,
}

#[uzu_config_abstract(HadamardSpec, KroneckerSpec)]
pub struct IncoherenceProcessorSpec;

#[uzu_config(self::IncoherenceProcessorSpec)]
pub struct HadamardSpec {
    pub block_size: u32,
    pub processing_mode: IncoherenceProcessingMode,
}

#[uzu_config(self::IncoherenceProcessorSpec)]
pub struct KroneckerSpec {
    pub processing_mode: IncoherenceProcessingMode,
}

#[uzu_config(super::WeightMatrixSpec)]
pub struct HybridSpec {
    pub quantization_spec: Box<AnyWeightMatrixSpec>,
    pub adapter_spec: Option<Unsupported>,
    pub incoherence_spec: Option<AnyIncoherenceProcessorSpec>,
}
