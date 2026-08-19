use proc_macros::uzu_config;

use crate::config::{
    linear::LinearConfig,
    normalization::{NormalizationConfig, UpcastMode},
};

#[uzu_config(super::TokenMixerConfig)]
pub struct AttentionConfig {
    #[serde(alias = "qkv_projection_config")]
    pub qkvg_projection_config: LinearConfig,
    pub out_projection_config: LinearConfig,

    pub query_norm_config: Option<NormalizationConfig>,
    pub key_norm_config: Option<NormalizationConfig>,

    pub num_heads: u32,
    pub num_groups: u32,
    pub head_dim: u32,
    pub is_causal: bool,
    pub scale: Option<f32>,
    pub sliding_window_size: Option<u32>,
    pub logit_soft_cap: Option<f32>,
    pub has_sinks: bool,
    #[serde(alias = "has_qkv_biases")]
    pub has_qkvg_biases: bool,
    pub has_out_biases: bool,
    /// Query-width sigmoid gate appended as the final fused projection segment.
    /// Older packs omit this flag and declare the gate through `gate_projection_config`.
    #[serde(default)]
    pub has_gate: bool,
    /// Legacy contract: a separate query-width gate projection. Presence implies a gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate_projection_config: Option<LinearConfig>,
    pub normalize_values: bool,
    pub is_kv_sharing: bool,
}

impl AttentionConfig {
    pub fn value_norm_config(&self) -> Option<NormalizationConfig> {
        self.normalize_values.then_some(NormalizationConfig {
            epsilon: 1e-6,
            scale_offset: None,
            upcast_mode: UpcastMode::FullLayer,
            subtract_mean: false,
            has_scale: false,
            has_biases: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use proc_macros::uzu_test;

    use super::AttentionConfig;

    const SHARED_FIELDS: &str = r#"
        "out_projection_config": {},
        "query_norm_config": null,
        "key_norm_config": null,
        "num_heads": 8,
        "num_groups": 2,
        "head_dim": 256,
        "is_causal": true,
        "scale": null,
        "sliding_window_size": null,
        "logit_soft_cap": null,
        "has_sinks": false,
        "has_out_biases": false,
        "normalize_values": false,
        "is_kv_sharing": false
    "#;

    #[uzu_test]
    fn parses_current_fused_contract() {
        let config: AttentionConfig = serde_json::from_str(&format!(
            r#"{{"type": "AttentionConfig", "qkvg_projection_config": {{}}, "has_qkvg_biases": true, "has_gate": true, {SHARED_FIELDS}}}"#
        ))
        .expect("current contract must parse");
        assert!(config.has_gate);
        assert!(config.has_qkvg_biases);
        assert_eq!(config.gate_projection_config, None);
    }

    #[uzu_test]
    fn parses_legacy_contract_with_separate_gate() {
        let config: AttentionConfig = serde_json::from_str(&format!(
            r#"{{"type": "AttentionConfig", "qkv_projection_config": {{}}, "has_qkv_biases": false, "gate_projection_config": {{}}, {SHARED_FIELDS}}}"#
        ))
        .expect("legacy gated contract must parse");
        assert!(!config.has_gate);
        assert!(config.gate_projection_config.is_some());
    }

    #[uzu_test]
    fn parses_legacy_contract_without_gate() {
        let config: AttentionConfig = serde_json::from_str(&format!(
            r#"{{"type": "AttentionConfig", "qkv_projection_config": {{}}, "has_qkv_biases": false, "gate_projection_config": null, {SHARED_FIELDS}}}"#
        ))
        .expect("legacy ungated contract must parse");
        assert!(!config.has_gate);
        assert_eq!(config.gate_projection_config, None);
    }

    #[uzu_test]
    fn serializes_only_the_current_contract() {
        let config: AttentionConfig = serde_json::from_str(&format!(
            r#"{{"type": "AttentionConfig", "qkv_projection_config": {{}}, "has_qkv_biases": true, "gate_projection_config": null, {SHARED_FIELDS}}}"#
        ))
        .expect("legacy contract must parse");
        let serialized = serde_json::to_value(&config).expect("config must serialize");
        let object = serialized.as_object().expect("config serializes to an object");
        assert!(object.contains_key("qkvg_projection_config"));
        assert!(object.contains_key("has_qkvg_biases"));
        assert!(object.contains_key("has_gate"));
        assert!(!object.contains_key("qkv_projection_config"));
        assert!(!object.contains_key("has_qkv_biases"));
        assert!(!object.contains_key("gate_projection_config"));
    }
}
