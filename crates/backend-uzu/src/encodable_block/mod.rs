mod activation;
mod attention;
mod classifier_layer;
mod decoder;
mod delta_net_mixer;
mod embedding;
mod layer;
mod layer_norm;
pub(crate) mod linear;
mod mamba_mixer;
pub(crate) mod mlp;
mod normalization;
mod pooling;
mod prediction_head;
mod qk_norm;
mod rms_norm;
mod rope;
mod sampling;
mod short_conv_mixer;
mod tensor_add_swap;
mod tensor_copy;

pub use activation::Activation;
pub use attention::Attention;
pub use classifier_layer::ClassifierLayer;
pub use decoder::Decoder;
pub(crate) use delta_net_mixer::DeltaNetMixer;
pub use embedding::Embedding;
pub use layer::LayerExecutables;
pub use layer_norm::{LayerNorm, LayerNormError};
pub use linear::{Linear, LoraFusion};
pub(crate) use mamba_mixer::MambaMixer;
pub use mlp::Mlp;
pub use normalization::Normalization;
pub use pooling::Pooling;
pub use prediction_head::ClassifierPredictionHead;
pub use qk_norm::QKNorm;
pub use rms_norm::{RMSNorm, RMSNormError};
pub use rope::Rope;
pub use sampling::Sampling;
pub use short_conv_mixer::ShortConvMixer;
pub use tensor_add_swap::TensorAddSwap;
pub use tensor_copy::TensorCopy;

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
