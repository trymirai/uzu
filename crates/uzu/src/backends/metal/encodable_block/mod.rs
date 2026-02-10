//! Encodable building blocks for classifier and LLM forward passes.
//!
//! These are higher-level abstractions that compose kernels into
//! reusable operations for both classifier and LLM pipelines.
//!
//! Encodables implement `EncodableBlock` and orchestrate one or more
//! kernels to perform operations on `ForwardPassState`.

pub use super::Metal;
pub use crate::encodable_block::EncodableBlock;

mod activation;
mod attention;
mod classifier_layer;
mod decoder;
mod embedding;
mod layer;
mod linear;
mod mamba_mixer;
mod mlp;
mod moe_block;
mod normalization;
mod pooling;
mod prediction_head;
mod rope;
mod short_conv_mixer;
mod tensor_add_swap;
mod tensor_copy;
pub mod transformer_layer;

pub use activation::Activation;
pub use attention::Attention;
pub use classifier_layer::ClassifierLayer;
pub use decoder::Decoder;
pub use embedding::{
    EmbeddingError, FullPrecisionEmbeddingLookup,
    FullPrecisionEmbeddingReadout, QuantizedEmbeddingLookup,
    QuantizedEmbeddingReadout,
};
pub use layer::LayerExecutables;
pub use linear::{FullPrecisionLinear, QuantizedLinear};
pub(crate) use mamba_mixer::MambaMixer;
pub use mlp::{MlpBlock, MlpFusedBlock, MlpFusedUpKernel};
pub use moe_block::{MoeBlock, SharedMoeWeights};
pub use normalization::{
    LayerNorm, Normalization, NormalizationError, QKNorm, RMSNorm,
};
pub use pooling::Pooling;
pub use prediction_head::ClassifierPredictionHead;
pub use rope::Rope;
pub(crate) use short_conv_mixer::ShortConvMixer;
pub use tensor_add_swap::TensorAddSwap;
pub use tensor_copy::TensorCopy;
