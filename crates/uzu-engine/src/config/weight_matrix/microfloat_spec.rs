use uzu_engine_macros::uzu_config;

use crate::{backends::common::microfloat::MicrofloatFormat, config::weight_matrix::Layout};

#[uzu_config(super::WeightMatrixSpec)]
pub struct MicrofloatSpec {
    pub bits: u32,
    pub group_size: usize,
    pub scale_mode: MicrofloatFormat,
    pub layout: Layout,
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use uzu_engine_macros::uzu_test;

    use super::*;
    use crate::config::weight_matrix::AnyWeightMatrixSpec;

    #[uzu_test]
    fn parses_mxfp4_scale_mode() {
        let spec: AnyWeightMatrixSpec = serde_json::from_value(json!({
            "type": "MicrofloatSpec",
            "bits": 4,
            "group_size": 16,
            "scale_mode": "mxfp4",
            "layout": "output_input"
        }))
        .unwrap();
        let AnyWeightMatrixSpec::MicrofloatSpec(spec) = spec else {
            panic!("expected MicrofloatSpec");
        };
        assert_eq!(spec.scale_mode, MicrofloatFormat::Mxfp4);
    }
}
