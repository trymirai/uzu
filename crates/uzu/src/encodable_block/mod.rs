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
pub struct EncodingParameters {
    pub projection_step: Option<usize>,
}

impl EncodingParameters {
    pub fn new() -> Self {
        Self {
            projection_step: None,
        }
    }

    pub fn with_projection(
        mut self,
        projection_step: usize,
    ) -> Self {
        self.projection_step = Some(projection_step);
        self
    }
}

pub trait EncodableBlock<B: Backend> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error>;
}
