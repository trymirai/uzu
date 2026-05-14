use serde::{Deserialize, Serialize};

use super::{AttentionConfig, EmbeddingConfig, TransformerConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DecoderConfig {
    pub embedding_config: EmbeddingConfig,
    pub transformer_config: TransformerConfig,
    pub vocab_size: usize,
    pub pard_token: Option<usize>,
}

impl DecoderConfig {
    pub fn first_attention(&self) -> Option<&AttentionConfig> {
        self.transformer_config.layer_configs.iter().find_map(|l| l.mixer_config.as_attention())
    }
}
