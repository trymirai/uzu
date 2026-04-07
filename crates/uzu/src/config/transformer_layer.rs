use serde::{Deserialize, Serialize};

use super::{AttentionConfig, MLPConfig, MixerConfig, NormalizationConfig};

/// Per-layer PLE config (lalamo PR #197 format).
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct PLELayerConfig {
    #[serde(default)]
    pub has_layer_scalar: bool,
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

    /// Per-layer MLP hidden dimension (lalamo PR #197 format)
    #[serde(default)]
    pub hidden_dim: Option<usize>,

    /// Source layer index for KV cache sharing (lalamo PR #197 format)
    #[serde(default)]
    pub kv_source_layer: Option<usize>,

    /// Per-layer PLE config (lalamo PR #197 format)
    #[serde(default)]
    pub ple_config: Option<PLELayerConfig>,
}

impl TransformerLayerConfig {
    pub fn attention_config(&self) -> Option<&AttentionConfig> {
        self.mixer_config.as_attention()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/config/transformer_layer_test.rs"]
mod tests;
