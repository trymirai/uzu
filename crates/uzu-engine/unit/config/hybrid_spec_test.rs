use uzu_engine_macros::uzu_test;

use super::{HybridSpec, IncoherenceProcessingMode};

fn hybrid(rotation: &str) -> Result<HybridSpec, serde_json::Error> {
    serde_json::from_str(&format!(
        r#"{{"type":"HybridSpec","quantization_spec":{{"type":"FullPrecisionSpec","layout":"output_input"}},"adapter_spec":null,{rotation}}}"#
    ))
}

#[uzu_test]
fn nested_rotation_parses() {
    let spec = hybrid(r#""incoherence_spec":{"type":"HadamardSpec","block_size":32,"processing_mode":"input_output"}"#)
        .unwrap();
    let rotation = spec.incoherence_spec.unwrap();
    assert_eq!(rotation.block_size, 32);
    assert_eq!(rotation.processing_mode, IncoherenceProcessingMode::InputOutput);
    assert!(hybrid(r#""incoherence_spec":null"#).unwrap().incoherence_spec.is_none());
}

#[uzu_test]
fn flat_rotation_fields_are_rejected() {
    // The format before lalamo#364; those models are re-exported, not read.
    assert!(hybrid(r#""incoherence_block_size":32,"incoherence_processing_mode":"input_output""#).is_err());
}

#[uzu_test]
fn kronecker_rotation_is_rejected() {
    assert!(hybrid(r#""incoherence_spec":{"type":"KroneckerSpec","block_size":32}"#).is_err());
}
