use serde::{Deserialize, Serialize};

use super::{NormalizationConfig, RoPEConfig, TransformerLayerConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct TransformerConfig {
    pub global_rope_config: Option<RoPEConfig>,
    pub local_rope_config: Option<RoPEConfig>,
    pub layer_configs: Vec<TransformerLayerConfig>,
    pub output_norm_config: NormalizationConfig,

    pub model_dim: usize,
    pub hidden_dim: usize,
    #[serde(default)]
    pub num_heads: Option<usize>,
    #[serde(default)]
    pub num_groups: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub attention_scale: Option<f32>,
    #[serde(default)]
    pub num_layers: Option<usize>,
    pub context_length: usize,

    /// Global attention RoPE dimension (lalamo PR #197). Used to derive partial_rope_dim.
    #[serde(default)]
    pub global_rope_dim: Option<usize>,

    /// Local attention RoPE dimension (lalamo PR #197). Deserialized for forward
    /// compatibility; not used in conversion because local layers use full head_dim for RoPE.
    #[serde(default)]
    pub local_rope_dim: Option<usize>,

    /// Global attention head dimension (lalamo PR #197). Deserialized for forward
    /// compatibility; per-layer head_dim from AttentionConfig is used instead.
    #[serde(default)]
    pub global_head_dim: Option<usize>,
}
