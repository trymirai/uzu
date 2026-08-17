use proc_macros::uzu_config;

use crate::config::weight_matrix::Layout;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum MicrofloatScaleMode {
    Mxfp4,
    Nvfp4,
}

#[uzu_config(super::WeightMatrixSpec)]
pub struct MicrofloatSpec {
    pub bits: u32,
    pub group_size: usize,
    pub scale_mode: MicrofloatScaleMode,
    pub layout: Layout,
}

#[cfg(test)]
mod tests {
    use proc_macros::uzu_test;
    use serde_json::json;

    use super::*;
    use crate::{
        backends::cpu::Cpu, config::weight_matrix::AnyWeightMatrixSpec, encodable_block::weight_matrix::parse_spec,
    };

    #[uzu_test]
    fn parses_mxfp4_without_aliasing_nvfp4() {
        for (mode, expected) in [("mxfp4", MicrofloatScaleMode::Mxfp4), ("nvfp4", MicrofloatScaleMode::Nvfp4)] {
            let spec: AnyWeightMatrixSpec = serde_json::from_value(json!({
                "type": "MicrofloatSpec",
                "bits": 4,
                "group_size": 16,
                "scale_mode": mode,
                "layout": "output_input"
            }))
            .unwrap();
            let AnyWeightMatrixSpec::MicrofloatSpec(spec) = spec else {
                panic!("expected MicrofloatSpec");
            };
            assert_eq!(spec.scale_mode, expected);
        }
    }

    #[uzu_test]
    fn rejects_unsupported_runtime_formats() {
        let spec: AnyWeightMatrixSpec = serde_json::from_value(json!({
            "type": "MicrofloatSpec",
            "bits": 4,
            "group_size": 16,
            "scale_mode": "nvfp4",
            "layout": "output_input"
        }))
        .unwrap();
        let error = match parse_spec::<Cpu>(&spec) {
            Ok(_) => panic!("NVFP4 runtime spec was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("NVFP4 runtime storage is not supported"));
    }
}
