// TODO: remove when implementing a consumer (expect warns once this is used).
#![expect(dead_code)]

use proc_macros::uzu_config;

use crate::config::{
    linear::LinearConfig, mlp::dense_mlp::DenseMLPConfig, normalization::NormalizationConfig, rope::AnyRoPEConfig,
};

#[uzu_config]
pub struct DFlashAttentionConfig {
    pub linear_config: LinearConfig,
    pub query_norm_config: NormalizationConfig,
    pub key_norm_config: NormalizationConfig,
    pub rope_config: AnyRoPEConfig,
    pub num_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub has_attention_biases: bool,
    pub has_output_biases: bool,
    /// Symmetric window: keys within +/- `sliding_window_size / 2` of the query
    /// position, not the causal sliding window used by uzu attention.
    pub sliding_window_size: Option<usize>,
    pub scale: f32,
}

#[uzu_config]
pub struct DFlashDraftLayerConfig {
    pub attention_config: DFlashAttentionConfig,
    pub input_norm_config: NormalizationConfig,
    /// Sits in the pre-MLP-norm position: normalizes hidden + attention output
    /// before the MLP, unlike uzu's post-mixer norm.
    pub post_attention_norm_config: NormalizationConfig,
    pub mlp_config: DenseMLPConfig,
}

#[uzu_config]
pub struct DFlashDraftConfig {
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub block_size: usize,
    pub mask_token_id: u64,
    pub target_layer_ids: Box<[usize]>,
    pub num_target_layers: usize,
    pub vocab_size: usize,
    pub context_projection_config: LinearConfig,
    pub context_norm_config: NormalizationConfig,
    pub layer_configs: Box<[DFlashDraftLayerConfig]>,
    pub output_norm_config: NormalizationConfig,
}
