use uzu_engine_macros::uzu_config;

use super::{AnyWeightMatrixSpec, HadamardSpec, IncoherenceProcessingMode};
use crate::utils::strict_serde::Unsupported;

#[uzu_config]
#[serde(untagged)]
pub enum HybridSpecInput {
    Nested {
        #[serde(rename = "type")]
        ty: monostate::MustBe!("HybridSpec"),
        quantization_spec: Box<AnyWeightMatrixSpec>,
        adapter_spec: Option<Unsupported>,
        incoherence_spec: Option<HadamardSpec>,
    },
    Flat {
        #[serde(rename = "type")]
        ty: monostate::MustBe!("HybridSpec"),
        quantization_spec: Box<AnyWeightMatrixSpec>,
        adapter_spec: Option<Unsupported>,
        incoherence_block_size: Option<u32>,
        incoherence_processing_mode: IncoherenceProcessingMode,
    },
}
