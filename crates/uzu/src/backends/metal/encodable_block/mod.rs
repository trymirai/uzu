//! Encodable building blocks for classifier and LLM forward passes.
//!
//! These are higher-level abstractions that compose kernels into
//! reusable operations for both classifier and LLM pipelines.
//!
//! Encodables implement `EncodableBlock` and orchestrate one or more
//! kernels to perform operations on `ForwardPassState`.

pub use super::Metal;
use super::forward_pass::ForwardPassState;
// Re-exports from generic implementation
pub use crate::encodable_block::activation;
pub use crate::encodable_block::{
    EncodableBlock, attention, classifier_layer, decoder, embedding,
};
mod encoding_parameters;
pub use activation::Activation;
pub use attention::Attention;
pub use classifier_layer::ClassifierLayer;
pub use decoder::Decoder;
pub use embedding::{
    EmbeddingError, FullPrecisionEmbeddingLookup,
    FullPrecisionEmbeddingReadout, QuantizedEmbeddingLookup,
    QuantizedEmbeddingReadout,
};
pub use encoding_parameters::EncodingParameters;
pub use layer::LayerExecutables;
pub use linear::{FullPrecisionLinear, QuantizedLinear};
pub(crate) use mamba_mixer::MambaMixer;
pub use mlp::{MlpBlock, MlpFusedBlock, MlpFusedUpKernel};
pub use moe_block::{MoeBlock, SharedMoeWeights};
pub use normalization::{
    LayerNorm, Normalization, NormalizationError, QKNorm, RMSNorm,
};
pub use pooling::Pooling;
// Check if ClassifierPredictionHead is exported from prediction_head module
pub use prediction_head::ClassifierPredictionHead;
pub use rope::Rope;
pub(crate) use short_conv_mixer::ShortConvMixer;
pub use tensor_add_swap::TensorAddSwap;
pub use tensor_copy::TensorCopy;

pub use crate::encodable_block::{
    layer, linear, mamba_mixer, mlp, moe_block, normalization, pooling,
    prediction_head, rope, short_conv_mixer, tensor_add_swap, tensor_copy,
    transformer_layer,
};
