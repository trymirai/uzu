use proc_macros::uzu_config;

use crate::config::{
    embedding::tied_embedding::TiedEmbeddingConfig, linear::LinearConfig, transformer::TransformerConfig,
};

#[uzu_config(super::TTSTextDecoderConfig)]
pub struct FishAudioTextDecoderConfig {
    pub slow_embeddings_config: TiedEmbeddingConfig,
    pub slow_model_config: TransformerConfig,
    pub slow_readout_config: LinearConfig,

    pub fast_embeddings_config: TiedEmbeddingConfig,
    pub fast_model_config: TransformerConfig,
    pub fast_readout_config: LinearConfig,

    pub codebook_embeddings_config: TiedEmbeddingConfig,
    pub fast_model_projection_config: Option<LinearConfig>,

    pub semantic_token_begin_id: u64,
    pub semantic_token_end_id: u64,
    pub im_end_token_id: u64,
    pub codebook_size: usize,
    pub vocab_size: usize,
    pub slow_model_dim: usize,
    pub fast_model_dim: usize,
    pub num_codebooks: usize,
    pub max_seq_len: usize,

    pub scale_codebook_embeddings: bool,

    pub short_logits_size: usize,
    pub repeat_window_size: usize,
}
