//! Encodable building blocks for classifier and LLM forward passes.
//!
//! These are higher-level abstractions that compose kernels into
//! reusable operations for both classifier and LLM pipelines.
//!
//! Encodables implement `EncodableBlock` and orchestrate one or more
//! kernels to perform operations on `ForwardPassState`.

use metal::ComputeCommandEncoderRef;
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::forward_pass::ForwardPassState;

mod attention;
mod classifier_layer;
mod decoder;
mod embedding;
mod encoding_parameters;
mod layer;
mod linear;
mod mamba_mixer;
mod mlp;
mod moe_block;
mod normalization;
mod pooling;
mod prediction_head;
mod rope;
mod sampling;
mod short_conv_mixer;
mod tensor_add_swap;
mod tensor_copy;
pub mod transformer_layer;

pub use attention::Attention;
pub use classifier_layer::ClassifierLayer;
pub use decoder::Decoder;
pub use embedding::{
    FullPrecisionEmbeddingLookup, FullPrecisionEmbeddingReadout,
    QuantizedEmbeddingError, QuantizedEmbeddingLookup, QuantizedEmbeddingReadout,
};
pub use encoding_parameters::EncodingParameters;
pub use layer::LayerExecutables;
pub use linear::{FullPrecisionLinear, QuantizedLinear};
pub(crate) use mamba_mixer::MambaMixer;
pub use mlp::MlpBlock;
pub use moe_block::{MoeBlock, SharedMoeWeights};
pub use normalization::{
    LayerNorm, Normalization, NormalizationError, QKNorm, RMSNorm,
};
pub use pooling::Pooling;
pub use prediction_head::ClassifierPredictionHead;
pub use rope::Rope;
pub use sampling::Sampling;
pub(crate) use short_conv_mixer::ShortConvMixer;
pub use tensor_add_swap::TensorAddSwap;
pub use tensor_copy::TensorCopy;

/// Trait for encodable blocks that operate on `ForwardPassState`.
pub trait EncodableBlock {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    );

    /// Returns true if this block supports using a shared compute encoder.
    fn supports_shared_encoder(&self) -> bool {
        false
    }

    /// Encode using a shared compute encoder. Only called if `supports_shared_encoder` returns true.
    fn encode_with_shared_encoder(
        &self,
        _state: &mut ForwardPassState,
        _encoder: &ComputeCommandEncoderRef,
        _parameters: &EncodingParameters,
    ) {
        panic!("encode_with_shared_encoder called on unsupported type");
    }
}
