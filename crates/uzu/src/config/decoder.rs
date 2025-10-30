use serde::{Deserialize, Serialize};

use super::{
    decoder_layer::DecoderLayerConfig, embedding::EmbeddingConfig,
    normalization::RMSNormConfig, rope::RoPEConfig,
};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecoderLayerType {
    Transformer,
    #[serde(rename = "ssm")]
    StateSpace {
        conv_dim: usize,
        kernel_size: usize,
        state_dim: usize,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DecoderConfig {
    pub embedding_config: EmbeddingConfig,
    pub global_rope_config: RoPEConfig,
    pub local_rope_config: Option<RoPEConfig>,
    pub layer_config: DecoderLayerConfig,
    pub output_norm_config: RMSNormConfig,

    pub vocab_size: usize,
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_groups: usize,
    pub head_dim: usize,
    pub attention_scale: Option<f32>,
    pub num_layers: usize,
    pub sliding_window_sizes: Option<Box<[Option<usize>]>>,
    pub layer_types: Option<Box<[DecoderLayerType]>>,
    pub context_length: usize,
}

impl DecoderConfig {
    pub fn group_size(&self) -> usize {
        self.num_heads * self.num_groups
    }
}

#[cfg(test)]
mod tests {
    use serde_json5::from_str;

    use super::{
        super::{
            attention::AttentionConfig,
            common::{Activation, ConfigDataType, QuantizationMode},
            embedding::{EmbeddingConfig, EmbeddingConfigCommon},
            linear::{LinearConfig, QuantizationConfig},
            mlp::{DenseMLPConfig, MLPConfig},
            normalization::UpcastMode,
            rope::RopeConfigCommon,
        },
        *,
    };

    #[test]
    fn test_decoder_config() {
        let config_str = r#"
            {
                "embedding_config": {
                    "type": "QuantizedTiedEmbeddingConfig",
                    "input_scale": null,
                    "logit_soft_cap": null,
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
                    "attention_config": {
                        "qkv_projection_config": {
                            "type": "QLoRALinearConfig",
                            "group_size": 32,
                            "weight_quantization_mode": "uint4",
                            "activation_quantization_mode": "int8",
                            "activation_precision": "bfloat16",
                            "lora_rank": 16,
                            "lora_scale": 2.0,
                        },
                        "out_projection_config": {
                            "type": "QLoRALinearConfig",
                            "group_size": 32,
                            "weight_quantization_mode": "uint4",
                            "activation_quantization_mode": "int8",
                            "activation_precision": "bfloat16",
                            "lora_rank": 16,
                            "lora_scale": 2.0,
                        },
                        "logit_soft_cap": null,
                "has_sinks": false,
                        "has_qkv_biases": false,
                        "has_out_biases": false
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
            embedding_config: EmbeddingConfig::QuantizedTied {
                common: EmbeddingConfigCommon {
                    input_scale: None,
                    logit_soft_cap: None,
                },
                embedding_quantization_mode: QuantizationMode::Int8,
                activation_quantization_mode: Some(QuantizationMode::Int8),
                activation_precision: ConfigDataType::BFloat16,
            },
            global_rope_config: RoPEConfig::Llama {
                common: RopeConfigCommon {
                    precision: ConfigDataType::BFloat16,
                    base: 500000.0,
                    max_sequence_length: 262144,
                },
                scaling_factor: 32.0,
                original_context_length: 8192,
                low_frequency_factor: 1.0,
                high_frequency_factor: 4.0,
            },
            local_rope_config: None,
            layer_config: DecoderLayerConfig {
                pre_attention_norm_config: RMSNormConfig {
                    scale_precision: ConfigDataType::BFloat16,
                    accumulation_precision: ConfigDataType::Float32,
                    epsilon: 1e-5,
                    scale_offset: None,
                    upcast_mode: UpcastMode::OnlyNormalization,
                },
                pre_mlp_norm_config: RMSNormConfig {
                    scale_precision: ConfigDataType::BFloat16,
                    accumulation_precision: ConfigDataType::Float32,
                    epsilon: 1e-5,
                    scale_offset: None,
                    upcast_mode: UpcastMode::OnlyNormalization,
                },
                attention_config: AttentionConfig {
                    qkv_projection_config: LinearConfig::QLoRA {
                        quantization: QuantizationConfig {
                            group_size: 32,
                            weight_quantization_mode: QuantizationMode::UInt4,
                            activation_quantization_mode: Some(
                                QuantizationMode::Int8,
                            ),
                            activation_precision: ConfigDataType::BFloat16,
                        },
                        lora_rank: 16,
                        lora_scale: 2.0,
                    },
                    out_projection_config: LinearConfig::QLoRA {
                        quantization: QuantizationConfig {
                            group_size: 32,
                            weight_quantization_mode: QuantizationMode::UInt4,
                            activation_quantization_mode: Some(
                                QuantizationMode::Int8,
                            ),
                            activation_precision: ConfigDataType::BFloat16,
                        },
                        lora_rank: 16,
                        lora_scale: 2.0,
                    },
                    query_norm_config: None,
                    key_norm_config: None,
                    logit_soft_cap: None,
                    has_sinks: false,
                    has_qkv_biases: false,
                    has_out_biases: false,
                },
                mlp_config: MLPConfig::Dense(DenseMLPConfig {
                    linear_config: LinearConfig::QLoRA {
                        quantization: QuantizationConfig {
                            group_size: 32,
                            weight_quantization_mode: QuantizationMode::UInt4,
                            activation_quantization_mode: Some(
                                QuantizationMode::Int8,
                            ),
                            activation_precision: ConfigDataType::BFloat16,
                        },
                        lora_rank: 16,
                        lora_scale: 2.0,
                    },
                    activation: Activation::SILU {
                        alpha: 1.0,
                    },
                }),
                post_attention_norm_config: None,
                post_mlp_norm_config: None,
            },
            output_norm_config: RMSNormConfig {
                scale_precision: ConfigDataType::BFloat16,
                accumulation_precision: ConfigDataType::Float32,
                epsilon: 1e-5,
                scale_offset: None,
                upcast_mode: UpcastMode::OnlyNormalization,
            },
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
}
