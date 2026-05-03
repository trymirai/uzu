use serde_json5::from_str;

use super::{
    super::{
        attention::AttentionConfig,
        common::ConfigDataType,
        embedding::{EmbeddingConfig, EmbeddingConfigCommon},
        error::ConfigError,
        language_model::InnerModelConfig,
        linear::{LinearConfig, QuantizationConfig},
        mlp::{DenseMLPConfig, MLPConfig},
        normalization::UpcastMode,
        rope::RopeConfigCommon,
    },
    *,
};
use crate::backends::common::{ActivationConfig, gpu_types::QuantizationMode};

#[test]
fn test_decoder_config() {
    let config_str = r#"
            {
                "embedding_config": {
                    "type": "MLXQuantizedTiedEmbeddingConfig",
                    "input_scale": null,
                    "logit_soft_cap": null,
                    "group_size": 128,
                    "embedding_quantization_mode": "int8",
                    "activation_quantization_mode": "int8",
                    "activation_precision": "bfloat16"
                },
                "global_rope_config": {
                    "type": "LlamaRoPEConfig",
                    "precision": "bfloat16",
                    "base": 500000.0,
                    "max_sequence_length": 262144,
                    "scaling_factor": 32.0,
                    "original_context_length": 8192,
                    "low_frequency_factor": 1.0,
                    "high_frequency_factor": 4.0
                },
                "local_rope_config": null,
                "layer_config": {
                    "pre_attention_norm_config": {
                        "scale_precision": "bfloat16",
                        "accumulation_precision": "float32",
                        "epsilon": 1e-05,
                        "scale_offset": null,
                        "upcast_mode": "only_normalization"
                    },
                    "mixer_config": {
                        "type": "AttentionConfig",
                        "qkv_projection_config": {
                            "type": "QLoRALinearConfig",
                            "group_size": 32,
                            "weight_quantization_mode": "uint4",
                            "activation_quantization_mode": "int8",
                            "activation_precision": "bfloat16",
                            "lora_rank": 16,
                            "lora_scale": 2.0
                        },
                        "out_projection_config": {
                            "type": "QLoRALinearConfig",
                            "group_size": 32,
                            "weight_quantization_mode": "uint4",
                            "activation_quantization_mode": "int8",
                            "activation_precision": "bfloat16",
                            "lora_rank": 16,
                            "lora_scale": 2.0
                        },
                        "query_norm_config": null,
                        "key_norm_config": null,
                        "num_heads": 12,
                        "num_groups": 12,
                        "head_dim": 64,
                        "is_causal": true,
                        "scale": null,
                        "sliding_window_size": null,
                        "logit_soft_cap": null,
                        "has_sinks": false,
                        "has_qkv_biases": false,
                        "has_out_biases": false,
                        "use_rope": true
                    },
                    "post_attention_norm_config": null,
                    "pre_mlp_norm_config": {
                        "scale_precision": "bfloat16",
                        "accumulation_precision": "float32",
                        "epsilon": 1e-05,
                        "scale_offset": null,
                        "upcast_mode": "only_normalization"
                    },
                    "mlp_config": {
                        "type": "DenseMLPConfig",
                        "linear_config": {
                            "type": "QLoRALinearConfig",
                            "group_size": 32,
                            "weight_quantization_mode": "uint4",
                            "activation_quantization_mode": "int8",
                            "activation_precision": "bfloat16",
                            "lora_rank": 16,
                            "lora_scale": 2.0
                        },
                        "activation": {"type": "SiLU"}
                    },
                    "post_mlp_norm_config": null
                },
                "output_norm_config": {
                    "scale_precision": "bfloat16",
                    "accumulation_precision": "float32",
                    "epsilon": 1e-05,
                    "scale_offset": null,
                    "upcast_mode": "only_normalization"
                },
                "vocab_size": 128256,
                "model_dim": 2048,
                "hidden_dim": 8192,
                "num_heads": 32,
                "num_groups": 8,
                "head_dim": 64,
                "attention_scale": null,
                "num_layers": 16,
                "sliding_window_sizes": null,
                "context_length": 8192
            }
        "#;

    let ground_truth_config = DecoderConfig {
        embedding_config: EmbeddingConfig::MLXQuantizedTied {
            common: EmbeddingConfigCommon {
                input_scale: None,
                logit_soft_cap: None,
            },
            group_size: 128,
            embedding_quantization_mode: QuantizationMode::INT8,
            activation_quantization_mode: Some(QuantizationMode::INT8),
            activation_precision: ConfigDataType::BFloat16,
        },
        global_rope_config: Some(RoPEConfig::Llama {
            common: RopeConfigCommon {
                precision: ConfigDataType::BFloat16,
                base: 500000.0,
                max_sequence_length: 262144,
                head_dim: None,
                partial_rotary_dim: None,
            },
            scaling_factor: 32.0,
            original_context_length: 8192,
            low_frequency_factor: 1.0,
            high_frequency_factor: 4.0,
        }),
        local_rope_config: None,
        layer_config: DecoderLayerConfig {
            pre_attention_norm_config: NormalizationConfig {
                scale_precision: ConfigDataType::BFloat16,
                accumulation_precision: ConfigDataType::Float32,
                epsilon: 1e-5,
                scale_offset: None,
                upcast_mode: UpcastMode::OnlyNormalization,
                subtract_mean: false,
                use_bias: false,
            },
            pre_mlp_norm_config: NormalizationConfig {
                scale_precision: ConfigDataType::BFloat16,
                accumulation_precision: ConfigDataType::Float32,
                epsilon: 1e-5,
                scale_offset: None,
                upcast_mode: UpcastMode::OnlyNormalization,
                subtract_mean: false,
                use_bias: false,
            },
            mixer_config: MixerConfig::Attention(AttentionConfig {
                qkv_projection_config: LinearConfig::QLoRA {
                    quantization: QuantizationConfig {
                        group_size: 32,
                        weight_quantization_mode: QuantizationMode::UINT4,
                        activation_quantization_mode: Some(QuantizationMode::INT8),
                        activation_precision: ConfigDataType::BFloat16,
                    },
                    lora_rank: 16,
                    lora_scale: 2.0,
                },
                out_projection_config: LinearConfig::QLoRA {
                    quantization: QuantizationConfig {
                        group_size: 32,
                        weight_quantization_mode: QuantizationMode::UINT4,
                        activation_quantization_mode: Some(QuantizationMode::INT8),
                        activation_precision: ConfigDataType::BFloat16,
                    },
                    lora_rank: 16,
                    lora_scale: 2.0,
                },
                query_norm_config: None,
                key_norm_config: None,
                num_heads: Some(12),
                num_groups: Some(12),
                head_dim: Some(64),
                is_causal: Some(true),
                scale: None,
                sliding_window_size: None,
                logit_soft_cap: None,
                has_sinks: false,
                has_qkv_biases: false,
                has_out_biases: false,
                has_gate: false,
                gate_projection_config: None,
                use_rope: true,
                partial_rope_dim: None,
            }),
            mlp_config: MLPConfig::Dense(DenseMLPConfig {
                linear_config: LinearConfig::QLoRA {
                    quantization: QuantizationConfig {
                        group_size: 32,
                        weight_quantization_mode: QuantizationMode::UINT4,
                        activation_quantization_mode: Some(QuantizationMode::INT8),
                        activation_precision: ConfigDataType::BFloat16,
                    },
                    lora_rank: 16,
                    lora_scale: 2.0,
                },
                activation: ActivationConfig::silu_default(),
                has_up_biases: false,
                has_down_biases: false,
                gate_clipping: None,
                up_clipping: None,
                activation_to_gate: true,
            }),
            post_attention_norm_config: None,
            post_mlp_norm_config: None,
            rope_config: None,
        },
        output_norm_config: NormalizationConfig {
            scale_precision: ConfigDataType::BFloat16,
            accumulation_precision: ConfigDataType::Float32,
            epsilon: 1e-5,
            scale_offset: None,
            upcast_mode: UpcastMode::OnlyNormalization,
            subtract_mean: false,
            use_bias: false,
        },
        layer_configs: None,
        vocab_size: 128256,
        model_dim: 2048,
        hidden_dim: 8192,
        num_heads: 32,
        num_groups: 8,
        head_dim: 64,
        attention_scale: None,
        num_layers: 16,
        sliding_window_sizes: None,
        layer_types: None,
        context_length: 8192,
    };

    let deserialized_config: DecoderConfig = from_str(config_str).unwrap();
    assert_eq!(deserialized_config, ground_truth_config);
}

#[test]
fn test_decoder_config_with_layer_configs() {
    let config_str = r#"
            {
                "embedding_config": {
                    "type": "TiedEmbeddingConfig",
                    "input_scale": null,
                    "logit_soft_cap": null,
                    "precision": "bfloat16"
                },
                "global_rope_config": null,
                "local_rope_config": null,
                "layer_configs": [
                    {
                        "pre_mixer_norm_config": {
                            "scale_precision": "bfloat16",
                            "accumulation_precision": "float32",
                            "epsilon": 1e-06,
                            "scale_offset": 1.0,
                            "upcast_mode": "full_layer"
                        },
                        "mixer_config": {
                            "type": "AttentionConfig",
                            "qkv_projection_config": {
                                "type": "FullPrecisionLinearConfig",
                                "precision": "bfloat16"
                            },
                            "out_projection_config": {
                                "type": "FullPrecisionLinearConfig",
                                "precision": "bfloat16"
                            },
                            "query_norm_config": null,
                            "key_norm_config": null,
                            "num_heads": 4,
                            "num_groups": 1,
                            "head_dim": 256,
                            "is_causal": true,
                            "scale": 0.0625,
                            "sliding_window_size": 512,
                            "logit_soft_cap": null,
                            "has_sinks": false,
                            "has_qkv_biases": false,
                            "has_out_biases": false,
                            "use_rope": true
                        },
                        "rope_config": {
                            "type": "UnscaledRoPEConfig",
                            "precision": "bfloat16",
                            "base": 10000.0,
                            "max_sequence_length": 32768,
                            "head_dim": 256
                        },
                        "post_mixer_norm_config": null,
                        "pre_mlp_norm_config": {
                            "scale_precision": "bfloat16",
                            "accumulation_precision": "float32",
                            "epsilon": 1e-06,
                            "scale_offset": 1.0,
                            "upcast_mode": "full_layer"
                        },
                        "mlp_config": {
                            "type": "DenseMLPConfig",
                            "linear_config": {
                                "type": "FullPrecisionLinearConfig",
                                "precision": "bfloat16"
                            },
                            "activation": {
                                "type": "GELU"
                            }
                        },
                        "post_mlp_norm_config": null
                    },
                    {
                        "pre_mixer_norm_config": {
                            "scale_precision": "bfloat16",
                            "accumulation_precision": "float32",
                            "epsilon": 1e-06,
                            "scale_offset": 1.0,
                            "upcast_mode": "full_layer"
                        },
                        "mixer_config": {
                            "type": "AttentionConfig",
                            "qkv_projection_config": {
                                "type": "FullPrecisionLinearConfig",
                                "precision": "bfloat16"
                            },
                            "out_projection_config": {
                                "type": "FullPrecisionLinearConfig",
                                "precision": "bfloat16"
                            },
                            "query_norm_config": null,
                            "key_norm_config": null,
                            "num_heads": 4,
                            "num_groups": 1,
                            "head_dim": 256,
                            "is_causal": true,
                            "scale": 0.0625,
                            "sliding_window_size": null,
                            "logit_soft_cap": null,
                            "has_sinks": false,
                            "has_qkv_biases": false,
                            "has_out_biases": false,
                            "use_rope": true
                        },
                        "rope_config": {
                            "type": "UnscaledRoPEConfig",
                            "precision": "bfloat16",
                            "base": 500000.0,
                            "max_sequence_length": 32768,
                            "head_dim": 256,
                            "partial_rotary_dim": 128
                        },
                        "post_mixer_norm_config": null,
                        "pre_mlp_norm_config": {
                            "scale_precision": "bfloat16",
                            "accumulation_precision": "float32",
                            "epsilon": 1e-06,
                            "scale_offset": 1.0,
                            "upcast_mode": "full_layer"
                        },
                        "mlp_config": {
                            "type": "DenseMLPConfig",
                            "linear_config": {
                                "type": "FullPrecisionLinearConfig",
                                "precision": "bfloat16"
                            },
                            "activation": {
                                "type": "GELU"
                            }
                        },
                        "post_mlp_norm_config": null
                    }
                ],
                "output_norm_config": {
                    "scale_precision": "bfloat16",
                    "accumulation_precision": "float32",
                    "epsilon": 1e-06,
                    "scale_offset": 1.0,
                    "upcast_mode": "full_layer"
                },
                "vocab_size": 32000,
                "model_dim": 1152,
                "hidden_dim": 6912,
                "context_length": 32768
            }
        "#;

    let config: DecoderConfig = from_str(config_str).unwrap();
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.num_heads, 4);
    assert_eq!(config.num_groups, 1);
    assert_eq!(config.head_dim, 256);
    assert_eq!(config.attention_scale, Some(0.0625));
    let sliding_windows = config.sliding_window_sizes.unwrap();
    assert_eq!(sliding_windows.len(), 2);
    assert_eq!(sliding_windows[0], Some(512));
    assert_eq!(sliding_windows[1], None);
    let local_rope = config.local_rope_config.as_ref().unwrap();
    assert_eq!(local_rope.common().base, 10000.0);
    assert_eq!(local_rope.common().head_dim, Some(256));
    let global_rope = config.global_rope_config.as_ref().unwrap();
    assert_eq!(global_rope.common().base, 500000.0);
    assert_eq!(global_rope.common().partial_rotary_dim, Some(128));
    let layer_types = config.layer_types.unwrap();
    assert!(matches!(layer_types[0], DecoderLayerType::Transformer));
    assert_eq!(config.layer_configs.unwrap().len(), 2);
}

#[test]
fn test_language_model_config_rejects_conflicting_per_layer_rope_configs() {
    let config = serde_json::from_value::<InnerModelConfig>(language_model_config_with_layer_rope_configs(&[
        (None, 10000.0, None),
        (None, 20000.0, None),
    ]))
    .unwrap();

    assert!(matches!(config.to_decoder_config(), Err(ConfigError::Invalid(_))));
}

#[test]
fn test_language_model_config_derives_global_and_local_rope_configs_from_layers() {
    let config = serde_json::from_value::<InnerModelConfig>(language_model_config_with_layer_rope_configs(&[
        (Some(512), 10000.0, None),
        (None, 500000.0, Some(128)),
    ]))
    .unwrap();

    let decoder_config = config.to_decoder_config().unwrap();
    let local_rope = decoder_config.local_rope_config.as_ref().unwrap();
    assert_eq!(local_rope.common().base, 10000.0);
    assert_eq!(local_rope.common().partial_rotary_dim, None);
    let global_rope = decoder_config.global_rope_config.as_ref().unwrap();
    assert_eq!(global_rope.common().base, 500000.0);
    assert_eq!(global_rope.common().partial_rotary_dim, Some(128));
}

fn language_model_config_with_layer_rope_configs(
    layer_rope_configs: &[(Option<usize>, f64, Option<usize>)]
) -> serde_json::Value {
    let linear_config = serde_json::json!({
        "type": "FullPrecisionLinearConfig",
        "precision": "bfloat16"
    });
    let norm_config = serde_json::json!({
        "scale_precision": "bfloat16",
        "accumulation_precision": "float32",
        "epsilon": 1e-06,
        "scale_offset": null,
        "upcast_mode": "full_layer"
    });
    let attention_config = serde_json::json!({
        "type": "AttentionConfig",
        "qkv_projection_config": linear_config.clone(),
        "out_projection_config": linear_config.clone(),
        "query_norm_config": null,
        "key_norm_config": null,
        "num_heads": 4,
        "num_groups": 1,
        "head_dim": 64,
        "is_causal": true,
        "scale": null,
        "sliding_window_size": null,
        "logit_soft_cap": null,
        "has_sinks": false,
        "has_qkv_biases": false,
        "has_out_biases": false,
        "use_rope": true
    });
    let layer_config = |sliding_window_size: Option<usize>, base: f64, partial_rotary_dim: Option<usize>| {
        let mut mixer_config = attention_config.clone();
        mixer_config["sliding_window_size"] = serde_json::json!(sliding_window_size);

        serde_json::json!({
            "pre_attention_norm_config": null,
            "mixer_config": mixer_config,
            "rope_config": {
                "type": "UnscaledRoPEConfig",
                "precision": "bfloat16",
                "base": base,
                "max_sequence_length": 1024,
                "head_dim": 64,
                "partial_rotary_dim": partial_rotary_dim
            },
            "post_attention_norm_config": null,
            "pre_mlp_norm_config": norm_config.clone(),
            "mlp_config": {
                "type": "DenseMLPConfig",
                "linear_config": linear_config.clone(),
                "activation": {
                    "type": "GELU"
                }
            },
            "post_mlp_norm_config": null
        })
    };
    let layer_configs = layer_rope_configs
        .iter()
        .map(|&(sliding_window_size, base, partial_rotary_dim)| {
            layer_config(sliding_window_size, base, partial_rotary_dim)
        })
        .collect::<Vec<_>>();

    serde_json::json!({
        "embedding_config": {
            "type": "UntiedEmbeddingConfig",
            "input_scale": null,
            "logit_soft_cap": null,
            "precision": "bfloat16"
        },
        "transformer_config": {
            "global_rope_config": null,
            "local_rope_config": null,
            "layer_configs": layer_configs,
            "output_norm_config": norm_config,
            "model_dim": 256,
            "hidden_dim": 512,
            "context_length": 1024
        },
        "vocab_size": 1024
    })
}
