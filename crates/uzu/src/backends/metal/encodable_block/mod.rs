//! Encodable building blocks for classifier and LLM forward passes.
//!
//! These are higher-level abstractions that compose kernels into
//! reusable operations for both classifier and LLM pipelines.
//!
//! Encodables implement `EncodableBlock` and orchestrate one or more
//! kernels to perform operations on `ForwardPassState`.

mod attention;
mod classifier_layer;
mod decoder;
mod embedding;
mod layer;
mod linear;
mod mamba_mixer;
mod mlp;
mod moe_block;
mod prediction_head;
mod short_conv_mixer;
pub mod transformer_layer;

pub use attention::Attention;
pub use classifier_layer::ClassifierLayer;
pub use decoder::Decoder;
pub use embedding::{
    EmbeddingError, FullPrecisionEmbeddingLookup, FullPrecisionEmbeddingReadout, QuantizedEmbeddingLookup,
    QuantizedEmbeddingReadout,
};
pub use layer::LayerExecutables;
pub use linear::{FullPrecisionLinear, QuantizedLinear};
pub(crate) use mamba_mixer::MambaMixer;
pub use mlp::MlpBlock;
pub use moe_block::{MoeBlock, SharedMoeWeights};
pub use prediction_head::ClassifierPredictionHead;
pub(crate) use short_conv_mixer::ShortConvMixer;
