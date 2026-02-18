//! Encodable building blocks for classifier and LLM forward passes.
//!
//! These are higher-level abstractions that compose kernels into
//! reusable operations for both classifier and LLM pipelines.
//!
//! Encodables implement `EncodableBlock` and orchestrate one or more
//! kernels to perform operations on `ForwardPassState`.

mod classifier_layer;
mod decoder;
mod embedding;
mod layer;
mod linear;
mod moe_block;
pub mod transformer_layer;

pub use classifier_layer::ClassifierLayer;
pub use decoder::Decoder;
pub use embedding::{
    EmbeddingError, FullPrecisionEmbeddingLookup, FullPrecisionEmbeddingReadout, QuantizedEmbeddingLookup,
    QuantizedEmbeddingReadout,
};
pub use layer::LayerExecutables;
pub use linear::{FullPrecisionLinear, QuantizedLinear};
pub use moe_block::{MoeBlock, SharedMoeWeights};
