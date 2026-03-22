use serde::{Deserialize, Serialize};

use super::{AttentionConfig, MLPConfig, MixerConfig, NormalizationConfig};

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
}

impl TransformerLayerConfig {
    pub fn attention_config(&self) -> Option<&AttentionConfig> {
        self.mixer_config.as_attention()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/config/transformer_layer_test.rs"]
mod tests;
