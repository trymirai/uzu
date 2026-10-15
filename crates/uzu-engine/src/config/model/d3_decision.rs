use std::collections::HashSet;

use uzu_engine_macros::uzu_config;

use crate::config::{
    linear::LinearConfig,
    normalization::{NormalizationConfig, UpcastMode},
    token_mixer::AnyTokenMixerConfig,
    transformer::TransformerConfig,
};

#[uzu_config]
pub struct D3DecisionMetadata {
    pub format_version: u32,
    pub format_id: String,
    pub prompt: String,
    pub base_model: String,
    pub revision: String,
    pub codes: Box<[String]>,
    pub token_ids: Box<[u32]>,
    pub temperature: f32,
    pub attention_mode: String,
    pub pooling: String,
    pub max_length: Option<u32>,
    pub readout_dtype: String,
}

// D3 consumes token ids and has no language-model token codec or generation head.
#[uzu_config]
pub struct D3DecisionModelConfig {
    #[serde(rename = "type")]
    ty: monostate::MustBe!("D3DecisionModelConfig"),
    pub transformer_config: TransformerConfig,
    pub readout_config: LinearConfig,
    pub decision: D3DecisionMetadata,
    pub vocab_size: u32,
    pub activation_dtype: String,
}

impl D3DecisionModelConfig {
    pub fn validate(&self) -> Result<(), String> {
        let decision = &self.decision;
        if decision.format_version != 1 || decision.format_id != "d3-code-readout-v1" || decision.prompt != "d3" {
            return Err("unsupported D3 format or prompt".into());
        }
        if self.activation_dtype != "bfloat16" || decision.readout_dtype != "float32" {
            return Err("D3 requires BF16 activations and an FP32 readout".into());
        }
        if decision.pooling != "last" || decision.attention_mode != "noncausal_full_attention" {
            return Err("D3 requires last-token pooling and noncausal full attention".into());
        }
        if !decision.temperature.is_finite() || decision.temperature <= 0.0 {
            return Err("D3 temperature must be finite and positive".into());
        }
        if decision.max_length == Some(0) || self.vocab_size == 0 {
            return Err("context and vocabulary bounds must be positive".into());
        }
        if self.transformer_config.model_dim != 4096
            || self.transformer_config.hidden_dim != 12288
            || decision.codes.len() != 255
        {
            return Err("this D3 adapter requires a 4096/12288-dimensional trunk and 255 readout codes".into());
        }
        if decision.token_ids.len() != decision.codes.len()
            || decision.codes.iter().any(String::is_empty)
            || decision.codes.iter().collect::<HashSet<_>>().len() != decision.codes.len()
            || decision.token_ids.iter().collect::<HashSet<_>>().len() != decision.token_ids.len()
            || decision.token_ids.iter().any(|token_id| *token_id >= self.vocab_size)
        {
            return Err("D3 codes and token ids must be unique, aligned, and within the vocabulary".into());
        }
        let mut attention_count = 0;
        let mut delta_count = 0;
        validate_rms(&self.transformer_config.output_norm_config, false, "output_norm")?;
        for (index, layer) in self.transformer_config.layer_configs.iter().enumerate() {
            if layer.ple_config.is_some() || layer.kv_source_layer_index.is_some() {
                return Err("D3 does not support per-layer embeddings or shared KV layers".into());
            }
            let pre_mixer_norm = layer
                .pre_mixer_norm_config
                .as_ref()
                .ok_or_else(|| format!("layer {index} is missing its pre-mixer RMS norm"))?;
            validate_rms(pre_mixer_norm, false, &format!("layer {index} pre_mixer_norm"))?;
            validate_rms(&layer.pre_mlp_norm_config, false, &format!("layer {index} pre_mlp_norm"))?;
            match &layer.mixer_config {
                AnyTokenMixerConfig::AttentionConfig(config) => {
                    if config.is_causal || config.sliding_window_size.is_some() {
                        return Err("all D3 full-attention layers must be noncausal without sliding windows".into());
                    }
                    if (config.num_heads, config.num_groups, config.head_dim) != (16, 4, 256) || config.is_kv_sharing {
                        return Err(format!(
                            "layer {index} has incompatible Qwen3.5 attention dimensions or KV sharing"
                        ));
                    }
                    let query_norm = config
                        .query_norm_config
                        .as_ref()
                        .ok_or_else(|| format!("layer {index} is missing its query RMS norm"))?;
                    let key_norm = config
                        .key_norm_config
                        .as_ref()
                        .ok_or_else(|| format!("layer {index} is missing its key RMS norm"))?;
                    validate_rms(query_norm, false, &format!("layer {index} query_norm"))?;
                    validate_rms(key_norm, false, &format!("layer {index} key_norm"))?;
                    attention_count += 1;
                },
                AnyTokenMixerConfig::DeltaNetConfig(config) => {
                    if (config.num_heads, config.num_groups, config.head_dim, config.value_head_dim, config.kernel_size)
                        != (32, 16, 128, 128, 4)
                    {
                        return Err(format!("layer {index} has incompatible Qwen3.5 DeltaNet dimensions"));
                    }
                    validate_rms(&config.norm_config, true, &format!("layer {index} gated_norm"))?;
                    delta_count += 1;
                },
                _ => return Err("D3 supports only Qwen3.5 attention and DeltaNet mixers".into()),
            }
        }
        if attention_count != 8 || delta_count != 24 {
            return Err("this D3 adapter requires 8 full-attention and 24 DeltaNet layers".into());
        }
        Ok(())
    }
}

fn validate_rms(
    config: &NormalizationConfig,
    gated: bool,
    path: &str,
) -> Result<(), String> {
    let (upcast_mode, scale_offset) = if gated {
        (UpcastMode::OnlyNormalization, None)
    } else {
        (UpcastMode::FullLayer, Some(1.0))
    };
    if config.upcast_mode != upcast_mode
        || config.scale_offset != scale_offset
        || !config.has_scale
        || config.has_biases
        || config.subtract_mean
        || config.epsilon != 1e-6
    {
        return Err(format!("{path} does not preserve the D3 Qwen3.5 RMS normalization contract"));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use serde_json::{Value, json};
    use uzu_engine_macros::uzu_test;

    use super::D3DecisionModelConfig;

    pub(crate) fn config_json() -> Value {
        let norm = json!({"epsilon": 1e-6, "scale_offset": 1.0, "upcast_mode": "full_layer",
            "subtract_mean": false, "has_scale": true, "has_biases": false});
        let gated_norm = json!({"epsilon": 1e-6, "scale_offset": null, "upcast_mode": "only_normalization",
            "subtract_mean": false, "has_scale": true, "has_biases": false});
        let layers = (0..32).map(|index| {
            let mixer = if index % 4 == 3 {
                json!({"type": "AttentionConfig", "qkvg_projection_config": {}, "out_projection_config": {},
                    "query_norm_config": norm.clone(), "key_norm_config": norm.clone(), "num_heads": 16, "num_groups": 4,
                    "head_dim": 256, "is_causal": false, "scale": null, "sliding_window_size": null,
                    "logit_soft_cap": null, "has_sinks": false, "has_qkvg_biases": false,
                    "has_out_biases": false, "has_gate": true, "normalize_values": false, "is_kv_sharing": false})
            } else {
                json!({"type": "DeltaNetConfig", "in_proj_config": {}, "conv_config": {"has_biases": false},
                    "out_proj_config": {}, "norm_config": gated_norm.clone(), "num_heads": 32, "num_groups": 16,
                    "head_dim": 128, "value_head_dim": 128, "kernel_size": 4})
            };
            json!({"pre_mixer_norm_config": norm.clone(), "mixer_conv_config": null, "mixer_config": mixer,
                "post_mixer_norm_config": null, "pre_mlp_norm_config": norm.clone(), "mlp_conv_config": null,
                "mlp_config": {"type": "DenseMLPConfig", "linear_config": {}, "activation": {"type": "SiLU", "alpha": 1.0},
                    "has_up_biases": false, "has_down_biases": false, "gate_clipping": null, "up_clipping": null},
                "post_mlp_norm_config": null, "hidden_dim": null, "ple_config": null, "has_post_layer_scalar": false,
                "kv_source_layer_index": null, "rope_config": null})
        }).collect::<Vec<_>>();
        json!({"type": "D3DecisionModelConfig", "activation_dtype": "bfloat16", "vocab_size": 256, "readout_config": {},
            "transformer_config": {"layer_configs": layers, "output_norm_config": norm, "model_dim": 4096, "hidden_dim": 12288},
            "decision": {"format_version": 1, "format_id": "d3-code-readout-v1", "prompt": "d3", "base_model": "Qwen/Qwen3.5-9B",
                "revision": "fixture", "codes": (0..255).map(|i| format!("code_{i}")).collect::<Vec<_>>(),
                "token_ids": (0u32..255).collect::<Vec<_>>(), "temperature": 1.0, "attention_mode": "noncausal_full_attention",
                "pooling": "last", "max_length": null, "readout_dtype": "float32"}})
    }

    #[uzu_test]
    fn d3_config_rejects_incompatible_numerics_and_attention() {
        let config: D3DecisionModelConfig = serde_json::from_value(config_json()).unwrap();
        assert!(config.validate().is_ok());
        let mut config = config.clone();
        config.activation_dtype = "float32".into();
        assert!(config.validate().is_err());
        let mut value = config_json();
        value["transformer_config"]["layer_configs"][3]["mixer_config"]["is_causal"] = json!(true);
        let config: D3DecisionModelConfig = serde_json::from_value(value).unwrap();
        assert!(config.validate().is_err());
    }

    #[uzu_test]
    fn d3_config_rejects_invalid_readout_mapping() {
        let mut config: D3DecisionModelConfig = serde_json::from_value(config_json()).unwrap();
        config.decision.token_ids[1] = config.decision.token_ids[0];
        assert!(config.validate().is_err());
        let mut config: D3DecisionModelConfig = serde_json::from_value(config_json()).unwrap();
        config.decision.temperature = f32::NAN;
        assert!(config.validate().is_err());
        let mut config: D3DecisionModelConfig = serde_json::from_value(config_json()).unwrap();
        config.decision.token_ids[0] = config.vocab_size;
        assert!(config.validate().is_err());
    }

    #[uzu_test]
    fn d3_config_schema_is_strict() {
        let mut value = config_json();
        value["decision"]["unexpected"] = json!(true);
        assert!(serde_json::from_value::<D3DecisionModelConfig>(value).is_err());
        let mut value = config_json();
        value["decision"].as_object_mut().unwrap().remove("max_length");
        assert!(serde_json::from_value::<D3DecisionModelConfig>(value).is_err());
        let mut value = config_json();
        value["type"] = json!("LanguageModelConfig");
        assert!(serde_json::from_value::<D3DecisionModelConfig>(value).is_err());
    }

    #[uzu_test]
    fn d3_config_rejects_lossy_rms_normalization() {
        for path in [
            "/transformer_config/output_norm_config/upcast_mode",
            "/transformer_config/layer_configs/0/pre_mixer_norm_config/upcast_mode",
            "/transformer_config/layer_configs/0/pre_mlp_norm_config/upcast_mode",
            "/transformer_config/layer_configs/3/mixer_config/query_norm_config/upcast_mode",
            "/transformer_config/layer_configs/3/mixer_config/key_norm_config/upcast_mode",
        ] {
            let mut value = config_json();
            *value.pointer_mut(path).unwrap() = json!("only_normalization");
            let config: D3DecisionModelConfig = serde_json::from_value(value).unwrap();
            assert!(config.validate().is_err(), "accepted stale RMS precision at {path}");
        }
        let mut value = config_json();
        value["transformer_config"]["layer_configs"][0]["mixer_config"]["norm_config"]["scale_offset"] = json!(1.0);
        let config: D3DecisionModelConfig = serde_json::from_value(value).unwrap();
        assert!(config.validate().is_err());
    }

    #[uzu_test]
    fn d3_config_preserves_fp32_readout_and_last_token_contract() {
        for (pointer, replacement) in [
            ("/decision/format_version", json!(2)),
            ("/decision/format_id", json!("language-model")),
            ("/decision/prompt", json!("chat")),
            ("/decision/readout_dtype", json!("bfloat16")),
            ("/decision/pooling", json!("mean")),
            ("/decision/attention_mode", json!("causal")),
            ("/decision/max_length", json!(0)),
            ("/vocab_size", json!(0)),
        ] {
            let mut value = config_json();
            *value.pointer_mut(pointer).unwrap() = replacement;
            let config: D3DecisionModelConfig = serde_json::from_value(value).unwrap();
            assert!(config.validate().is_err(), "accepted incompatible contract at {pointer}");
        }
    }

    #[uzu_test]
    fn d3_config_accepts_explicit_context_bounds_without_a_1024_token_cap() {
        for bound in [None, Some(1), Some(1024), Some(1025), Some(2048)] {
            let mut config: D3DecisionModelConfig = serde_json::from_value(config_json()).unwrap();
            config.decision.max_length = bound;
            assert!(config.validate().is_ok(), "rejected valid context bound {bound:?}");
        }
        for temperature in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut config: D3DecisionModelConfig = serde_json::from_value(config_json()).unwrap();
            config.decision.temperature = temperature;
            assert!(config.validate().is_err(), "accepted invalid temperature {temperature}");
        }
    }

    #[uzu_test]
    fn d3_config_rejects_missing_duplicate_or_empty_answer_codes() {
        for mutation in ["count", "empty", "duplicate", "unaligned"] {
            let mut value = config_json();
            match mutation {
                "count" => {
                    value["decision"]["codes"].as_array_mut().unwrap().pop();
                },
                "empty" => value["decision"]["codes"][0] = json!(""),
                "duplicate" => value["decision"]["codes"][1] = value["decision"]["codes"][0].clone(),
                "unaligned" => {
                    value["decision"]["token_ids"].as_array_mut().unwrap().pop();
                },
                _ => unreachable!(),
            }
            let config: D3DecisionModelConfig = serde_json::from_value(value).unwrap();
            assert!(config.validate().is_err(), "accepted {mutation} code mapping");
        }
    }

    #[uzu_test]
    fn d3_config_rejects_incompatible_backbone_and_mixer_shapes() {
        for (pointer, replacement) in [
            ("/transformer_config/model_dim", json!(2048)),
            ("/transformer_config/hidden_dim", json!(4096)),
            ("/transformer_config/layer_configs/0/pre_mixer_norm_config", Value::Null),
            ("/transformer_config/layer_configs/0/kv_source_layer_index", json!(0)),
            ("/transformer_config/layer_configs/3/mixer_config/sliding_window_size", json!(1024)),
            ("/transformer_config/layer_configs/3/mixer_config/num_heads", json!(8)),
            ("/transformer_config/layer_configs/3/mixer_config/num_groups", json!(2)),
            ("/transformer_config/layer_configs/3/mixer_config/head_dim", json!(128)),
            ("/transformer_config/layer_configs/3/mixer_config/is_kv_sharing", json!(true)),
            ("/transformer_config/layer_configs/3/mixer_config/query_norm_config", Value::Null),
            ("/transformer_config/layer_configs/3/mixer_config/key_norm_config", Value::Null),
            ("/transformer_config/layer_configs/0/mixer_config/num_heads", json!(16)),
            ("/transformer_config/layer_configs/0/mixer_config/num_groups", json!(8)),
            ("/transformer_config/layer_configs/0/mixer_config/head_dim", json!(64)),
            ("/transformer_config/layer_configs/0/mixer_config/value_head_dim", json!(64)),
            ("/transformer_config/layer_configs/0/mixer_config/kernel_size", json!(3)),
        ] {
            let mut value = config_json();
            *value.pointer_mut(pointer).unwrap() = replacement;
            let config: D3DecisionModelConfig = serde_json::from_value(value).unwrap();
            assert!(config.validate().is_err(), "accepted incompatible layer shape at {pointer}");
        }
        let mut value = config_json();
        value["transformer_config"]["layer_configs"].as_array_mut().unwrap().pop();
        let config: D3DecisionModelConfig = serde_json::from_value(value).unwrap();
        assert!(config.validate().is_err(), "accepted missing full-attention layer");
    }

    #[uzu_test]
    fn d3_config_validates_rms_scale_epsilon_and_bias_semantics() {
        for prefix in [
            "/transformer_config/output_norm_config",
            "/transformer_config/layer_configs/0/pre_mixer_norm_config",
            "/transformer_config/layer_configs/0/pre_mlp_norm_config",
            "/transformer_config/layer_configs/3/mixer_config/query_norm_config",
            "/transformer_config/layer_configs/3/mixer_config/key_norm_config",
            "/transformer_config/layer_configs/0/mixer_config/norm_config",
        ] {
            for (field, replacement) in [
                ("epsilon", json!(2e-6)),
                ("has_scale", json!(false)),
                ("has_biases", json!(true)),
                ("subtract_mean", json!(true)),
            ] {
                let mut value = config_json();
                *value.pointer_mut(&format!("{prefix}/{field}")).unwrap() = replacement;
                let config: D3DecisionModelConfig = serde_json::from_value(value).unwrap();
                assert!(config.validate().is_err(), "accepted incompatible RMS {prefix}/{field}");
            }
        }
    }
}
