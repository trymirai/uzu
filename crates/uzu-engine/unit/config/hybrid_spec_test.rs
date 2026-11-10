use uzu_engine_macros::uzu_test;

use crate::config::weight_matrix::AnyWeightMatrixSpec;

fn hybrid(rotation: &str) -> Result<AnyWeightMatrixSpec, serde_json::Error> {
    serde_json::from_str(&format!(
        r#"{{"type":"HybridSpec","quantization_spec":{{"type":"IntSpec","bits":4,"group_size":32,"is_symmetric":false,"layout":"input_output"}},"adapter_spec":null,{rotation}}}"#
    ))
}

#[uzu_test]
fn nested_rotation_parses() {
    for mode in ["input", "output", "input_output"] {
        let spec = hybrid(&format!(
            r#""incoherence_spec":{{"type":"HadamardSpec","block_size":32,"processing_mode":"{mode}"}}"#
        ))
        .unwrap();
        let AnyWeightMatrixSpec::HybridSpec(spec) = spec else {
            panic!("expected hybrid weights");
        };
        let rotation = spec.incoherence_spec.unwrap();
        assert_eq!(rotation.block_size, 32);
        assert_eq!(serde_json::to_value(rotation.processing_mode).unwrap(), mode);
        assert!(matches!(*spec.quantization_spec, AnyWeightMatrixSpec::IntSpec(_)));
    }
    let AnyWeightMatrixSpec::HybridSpec(spec) = hybrid(r#""incoherence_spec":null"#).unwrap() else {
        panic!("expected hybrid weights");
    };
    assert!(spec.incoherence_spec.is_none());
}

#[uzu_test]
fn unsupported_rotations_are_rejected() {
    assert!(
        hybrid(r#""incoherence_spec":null,"incoherence_block_size":32,"incoherence_processing_mode":"input_output""#)
            .is_err()
    );
    assert!(hybrid(r#""incoherence_block_size":32"#).is_err());
    assert!(hybrid(r#""incoherence_spec":{"type":"KroneckerSpec","block_size":32}"#).is_err());
    assert!(
        hybrid(r#""incoherence_spec":{"type":"HadamardSpec","block_size":32,"processing_mode":"unknown"}"#).is_err()
    );
}

#[uzu_test]
fn flat_rotation_normalizes_to_nested() {
    for mode in ["input", "output", "input_output"] {
        let flat = hybrid(&format!(r#""incoherence_block_size":32,"incoherence_processing_mode":"{mode}""#)).unwrap();
        let nested = hybrid(&format!(
            r#""incoherence_spec":{{"type":"HadamardSpec","block_size":32,"processing_mode":"{mode}"}}"#
        ))
        .unwrap();
        assert_eq!(flat, nested);
        assert_eq!(serde_json::to_value(flat).unwrap(), serde_json::to_value(nested).unwrap());
    }
}
