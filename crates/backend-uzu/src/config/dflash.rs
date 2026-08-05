use proc_macros::uzu_config;

use crate::config::{
    linear::LinearConfig, mlp::dense_mlp::DenseMLPConfig, normalization::NormalizationConfig, rope::AnyRoPEConfig,
    token_mixer::attention::AttentionConfig,
};

#[uzu_config]
pub struct DFlashDraftLayerConfig {
    pub attention_config: AttentionConfig,
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
    pub rope_config: AnyRoPEConfig,
    pub layer_configs: Box<[DFlashDraftLayerConfig]>,
    pub output_norm_config: NormalizationConfig,
}
