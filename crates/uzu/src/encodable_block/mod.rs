use crate::backends::{
    common::{Backend, CommandBuffer},
    metal::ForwardPassState,
};

pub use crate::backends::metal::Metal;
pub use crate::backends::metal::ForwardPassState;

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
mod sampling;
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
pub use prediction_head::PredictionHead;
pub use rope::Rope;
pub(crate) use short_conv_mixer::ShortConvMixer;
pub use sampling::Sampling;
pub use tensor_add_swap::TensorAddSwap;
pub use tensor_copy::TensorCopy;
pub use transformer_layer::TransformerLayer;

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
        state: &mut ForwardPassState,
        encoder: &B::EncoderRef,
        parameters: &EncodingParameters<B>,
    );

    fn supports_shared_encoder(&self) -> bool {
        false
    }

    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &B::CommandBuffer,
        parameters: &EncodingParameters<B>,
    ) {
        command_buffer.with_encoder(|encoder| {
            self.encode_with_shared_encoder(state, encoder, parameters)
        });
    }
}
