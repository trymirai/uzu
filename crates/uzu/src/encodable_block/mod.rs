use crate::{
    backends::common::{Backend, CommandBuffer},
    forward_pass::state::ForwardPassState,
};

mod activation;
mod attention;
mod classifier_layer;
mod decoder;
mod full_precision_embedding_lookup;
mod full_precision_embedding_readout;
mod full_precision_linear;
mod layer;
mod layer_norm;
mod mamba_mixer;
mod mlp;
mod moe_block;
mod normalization;
mod pooling;
mod prediction_head;
mod qk_norm;
mod quantized_embedding_lookup;
mod quantized_embedding_readout;
mod quantized_linear;
mod rms_norm;
mod rope;
mod sampling;
mod short_conv_mixer;
mod tensor_add_swap;
mod tensor_copy;
mod transformer_layer;

pub use activation::Activation;
pub use attention::Attention;
pub use classifier_layer::ClassifierLayer;
pub use decoder::Decoder;
pub use full_precision_embedding_lookup::{FullPrecisionEmbeddingLookup, FullPrecisionEmbeddingLookupError};
pub use full_precision_embedding_readout::{FullPrecisionEmbeddingReadout, FullPrecisionEmbeddingReadoutError};
pub use full_precision_linear::{FullPrecisionLinear, FullPrecisionLinearError};
pub use layer::LayerExecutables;
pub use layer_norm::{LayerNorm, LayerNormError};
pub(crate) use mamba_mixer::MambaMixer;
pub use mlp::MlpBlock;
pub use moe_block::MoeBlock;
pub use normalization::{Normalization, NormalizationError};
pub use pooling::Pooling;
pub use prediction_head::ClassifierPredictionHead;
pub use qk_norm::{QKNorm, QKNormError};
pub use quantized_embedding_lookup::{QuantizedEmbeddingLookup, QuantizedEmbeddingLookupError};
pub use quantized_embedding_readout::{QuantizedEmbeddingReadout, QuantizedEmbeddingReadoutError};
pub use quantized_linear::{QuantizedLinear, QuantizedLinearError};
pub use rms_norm::{RMSNorm, RMSNormError};
pub use rope::Rope;
pub use sampling::Sampling;
pub use short_conv_mixer::ShortConvMixer;
pub use tensor_add_swap::TensorAddSwap;
pub use tensor_copy::TensorCopy;
pub use transformer_layer::{embed_block, linear_block};

#[derive(Clone)]
pub struct EncodingParameters<'a, B: Backend> {
    pub warmup: bool,
    pub enable_commit: bool,
    pub wait_until_completed: bool,
    pub projection_step: Option<usize>,
    pub predicate: Option<&'a B::NativeBuffer>,
}

impl<'a, B: Backend> EncodingParameters<'a, B> {
    pub fn new(
        warmup: bool,
        enable_commit: bool,
        wait_until_completed: bool,
    ) -> Self {
        Self {
            warmup,
            enable_commit,
            wait_until_completed,
            projection_step: None,
            predicate: None,
        }
    }

    pub fn with_projection(
        mut self,
        projection_step: usize,
    ) -> Self {
        self.projection_step = Some(projection_step);
        self
    }

    pub fn with_predicate(
        mut self,
        predicate: &'a B::NativeBuffer,
    ) -> Self {
        self.predicate = Some(predicate);
        self
    }

    pub fn predicate_ref(&self) -> Option<&B::NativeBuffer> {
        self.predicate
    }
}

pub trait EncodableBlock<B: Backend> {
    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
    );

    fn supports_shared_encoder(&self) -> bool {
        false
    }

    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters<B>,
        command_buffer: &B::CommandBuffer,
    ) {
        command_buffer.with_compute_encoder(|encoder| self.encode_with_shared_encoder(state, parameters, encoder));
    }
}
