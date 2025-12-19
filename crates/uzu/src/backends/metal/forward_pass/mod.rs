use super::MTLContext;

mod encoder_resolver;
mod helpers;

// Re-export EncoderResolver (Metal specific)
pub use encoder_resolver::EncoderResolver;
pub use helpers::{
    encode_copy_array, update_cache_layers_after_acceptance,
    update_kv_cache_layer_after_acceptance,
};

// Import generic types
use crate::forward_pass;

// Type Aliases for Metal Backend
pub type ForwardPassState = forward_pass::ForwardPassState<MTLContext>;
pub type CacheLayers = forward_pass::CacheLayers<MTLContext>;
pub type CacheLayer = forward_pass::CacheLayer<MTLContext>;
pub type KVCacheLayer = forward_pass::KVCacheLayer<MTLContext>;
pub type ShortConvLayer = forward_pass::ShortConvLayer<MTLContext>;
pub type SSMLayer = forward_pass::SSMLayer<MTLContext>;
pub type ScratchBuffers = forward_pass::ScratchBuffers<MTLContext>;
pub type SharedBuffers = forward_pass::SharedBuffers<MTLContext>;
pub type CommonAuxBuffers = forward_pass::CommonAuxBuffers<MTLContext>;
pub type LanguageModelGeneratorAuxBuffers =
    forward_pass::LanguageModelGeneratorAuxBuffers<MTLContext>;
pub type EmbeddingsBuffers = forward_pass::EmbeddingsBuffers<MTLContext>;
pub type RopeBuffers = forward_pass::RopeBuffers<MTLContext>;
pub type MoeExpertWeights = forward_pass::MoeExpertWeights<MTLContext>;
pub type ClassifierModeState = forward_pass::ClassifierModeState<MTLContext>;
pub type LanguageModelGeneratorModeState =
    forward_pass::LanguageModelGeneratorModeState<MTLContext>;
pub type ForwardPassMode = forward_pass::ForwardPassMode<MTLContext>;
pub type ArrayCell = forward_pass::ArrayCell<MTLContext>;

// Re-export common types that don't need specialization or are enums/constants
pub use forward_pass::{
    ArrayId, AttentionBiasUpdate, HashMapId, INVALID_POSITION,
    KVCacheLayerState, ModelShape, RopeType,
};

#[cfg(feature = "tracing")]
pub type ActivationTrace = forward_pass::traces::ActivationTrace<MTLContext>;

pub use super::encodable_block::{EncodableBlock, EncodingParameters};
