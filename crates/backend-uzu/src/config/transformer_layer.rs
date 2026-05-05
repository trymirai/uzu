use serde::{Deserialize, Serialize};

use super::{AttentionConfig, LinearConfig, MLPConfig, MixerConfig, NormalizationConfig, RoPEConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct PLELayerConfig {
    pub linear_config: LinearConfig,
    pub norm_config: NormalizationConfig,
    pub ple_dim: usize,
    pub activation: crate::backends::common::ActivationConfig,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct PLEModelConfig {
    pub ple_dim: usize,
    pub num_layers: usize,
    pub ple_vocab_size: usize,
    pub ple_embed_scale: f32,
    pub model_projection_scale: f32,
    pub input_scale: f32,
    pub linear_config: LinearConfig,
    pub norm_config: NormalizationConfig,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct TransformerLayerConfig {
    #[serde(alias = "pre_mixer_norm_config")]
    pub pre_attention_norm_config: Option<NormalizationConfig>,
    pub mixer_config: MixerConfig,
    #[serde(alias = "post_mixer_norm_config")]
    pub post_attention_norm_config: Option<NormalizationConfig>,
    pub pre_mlp_norm_config: NormalizationConfig,
    pub mlp_config: MLPConfig,
    pub post_mlp_norm_config: Option<NormalizationConfig>,
    #[serde(default)]
    pub hidden_dim: Option<usize>,
    #[serde(default)]
    pub ple_config: Option<PLELayerConfig>,
    #[serde(default)]
    pub has_post_layer_scalar: bool,
    #[serde(default)]
    pub kv_source_layer: Option<usize>,
    #[serde(default)]
    pub rope_config: Option<RoPEConfig>,
}

impl TransformerLayerConfig {
    pub fn attention_config(&self) -> Option<&AttentionConfig> {
        self.mixer_config.as_attention()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/config/transformer_layer_test.rs"]
mod tests;
