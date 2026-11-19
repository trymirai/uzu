use std::range::Range;

use shoji::types::session::vision::VisionImageInput;
use thiserror::Error;

use crate::{
    backends::{
        Backend,
        common::{CommandBuffer, CommandBufferEncoding},
    },
    encodable_block::{linear::LinearEncodeError, mlp::MlpEncodeError, vision_layer::VisionLayerEncodeError},
    engine::language_model::vision::model::VisionModel,
};

pub struct VisionPrefilledImage<B: Backend> {
    embeddings: B::ScratchBuffer,
    width: u32,
    height: u32,
}

pub struct VisionPrefillResult<B: Backend> {
    images: Box<[VisionPrefilledImage<B>]>,
}

#[derive(Debug, Error)]
pub enum VisionEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Vision layer error: {0}")]
    Layer(#[from] VisionLayerEncodeError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearEncodeError<B>),
    #[error("MLP error: {0}")]
    Mlp(#[from] MlpEncodeError<B>),
}

#[derive(Debug, Error)]
pub enum VisionPrefillError<B: Backend> {
    #[error("Vision encoder error: {0}")]
    Encode(#[from] VisionEncodeError<B>),
    #[error("Image tokens {tokens:?} overlap the previous image or exceed the input of {input_length} tokens")]
    InvalidTokens {
        tokens: Range<usize>,
        input_length: usize,
    },
    #[error("Model has no vision encoder")]
    NoVisionEncoder,
    #[error("Unsupported patch grid {0:?}")]
    UnsupportedGrid([u32; 3]),
}

impl<B: Backend> VisionModel<B> {
    pub fn prefill(
        &self,
        images: &[VisionImageInput],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<VisionPrefillResult<B>, VisionPrefillError<B>> {
        let mut images = Vec::<VisionPrefilledImage<B>>::with_capacity(images.len());
        let result = VisionPrefillResult {
            images: images.into_boxed_slice(),
        };
        Ok(result)
    }

    fn encode(
        &self,
        image: &VisionImageInput,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, VisionEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("vision encode");
        let result = command_buffer.allocate_scratch(0).map_err(VisionEncodeError::Backend)?;
        Ok(result)
    }
}
