use uzu_engine_macros::uzu_config;

use crate::config::{
    linear::LinearConfig, normalization::NormalizationConfig, rope::AnyRoPEConfig,
    transformer_layer::TransformerLayerConfig,
};

/// A softmax-weighted sum of the target's hidden-state taps, RMS-normalized.
#[uzu_config]
pub struct TapMixConfig {
    pub num_taps: u32,
    pub norm_config: NormalizationConfig,
}

/// A draft layer whose attention also reads keys and values projected from a mix of the target's taps.
#[uzu_config]
pub struct DSparkLayerConfig {
    pub transformer_layer_config: TransformerLayerConfig,
    pub context_mix_config: TapMixConfig,
    /// Keys, then values, of each context row.
    pub context_kv_projection_config: LinearConfig,
    pub context_key_norm_config: NormalizationConfig,
}

/// A recurrent head that adds a vocabulary bias to each proposal position's logits, left to right.
#[uzu_config]
pub struct MarkovHeadConfig {
    pub rank: u32,
    pub vocab_size: u32,
    pub context_mix_config: TapMixConfig,
    pub recurrence_config: LinearConfig,
    pub output_config: LinearConfig,
    pub logit_soft_cap: Option<f32>,
}

/// A block drafter whose anchor and mask positions attend to each other and to the newest `context_window` context
/// rows. `target_feature_indices` are target layer visits whose incoming hidden states it taps; the index equal to
/// the number of visits is the final normalized hidden state.
#[uzu_config(super::SpeculatorConfig)]
pub struct DSparkSpeculatorConfig {
    pub model_dim: u32,
    pub hidden_dim: u32,
    pub block_size: u32,
    pub target_feature_indices: Box<[u32]>,
    pub context_window: u32,
    pub rope_config: AnyRoPEConfig,
    pub normalizes_anchor_embedding: bool,
    pub layer_configs: Box<[DSparkLayerConfig]>,
    pub output_norm_config: NormalizationConfig,
    pub markov_head_config: MarkovHeadConfig,
}
