use uzu_engine_macros::uzu_config;

use crate::{
    config::weight_matrix::{AnyWeightMatrixSpec, HadamardSpec, HybridSpecInput},
    utils::strict_serde::Unsupported,
};

#[uzu_config(super::WeightMatrixSpec)]
#[serde(from = "HybridSpecInput")]
pub struct HybridSpec {
    pub quantization_spec: Box<AnyWeightMatrixSpec>,
    pub adapter_spec: Option<Unsupported>,
    pub incoherence_spec: Option<HadamardSpec>,
}

impl From<HybridSpecInput> for HybridSpec {
    fn from(input: HybridSpecInput) -> Self {
        let (ty, quantization_spec, adapter_spec, incoherence_spec) = match input {
            HybridSpecInput::Nested {
                ty,
                quantization_spec,
                adapter_spec,
                incoherence_spec,
            } => (ty, quantization_spec, adapter_spec, incoherence_spec),
            HybridSpecInput::Flat {
                ty,
                quantization_spec,
                adapter_spec,
                incoherence_block_size,
                incoherence_processing_mode,
            } => (
                ty,
                quantization_spec,
                adapter_spec,
                incoherence_block_size.map(|block_size| HadamardSpec::new(block_size, incoherence_processing_mode)),
            ),
        };
        Self {
            ty,
            quantization_spec,
            adapter_spec,
            incoherence_spec,
        }
    }
}

#[cfg(test)]
#[path = "../../../unit/config/hybrid_spec_test.rs"]
mod tests;
