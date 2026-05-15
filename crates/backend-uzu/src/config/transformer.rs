use serde::{Deserialize, Serialize};

use super::{NormalizationConfig, TransformerLayerConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct TransformerConfig {
    pub layer_configs: Vec<TransformerLayerConfig>,
    pub output_norm_config: NormalizationConfig,
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub context_length: usize,
}
