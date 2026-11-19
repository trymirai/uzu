use shoji::types::session::vision::VisionImageInput;
use thiserror::Error;

use crate::{
    backends::{
        Backend,
        common::{CommandBuffer, CommandBufferEncoding, kernel::VisionPatchifyKernel},
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
    #[error("Model has no vision encoder")]
    NoVisionEncoder,
}

impl<B: Backend> VisionModel<B> {
    pub fn prefill(
        &self,
        images: &[VisionImageInput],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<VisionPrefillResult<B>, VisionPrefillError<B>> {
        let patch_size = self.config.encoder_config.patch_size;
        let merge_size = self.config.encoder_config.spatial_merge_size;

        let mut prefilled_images = Vec::<VisionPrefilledImage<B>>::with_capacity(images.len());
        for image in images {
            let embeddings = self.encode(image, command_buffer)?;
            let prefilled_image = VisionPrefilledImage {
                embeddings,
                width: image.width / patch_size / merge_size,
                height: image.height / patch_size / merge_size,
            };
            prefilled_images.push(prefilled_image);
        }

        let result = VisionPrefillResult {
            images: prefilled_images.into_boxed_slice(),
        };
        Ok(result)
    }

    fn encode(
        &self,
        image: &VisionImageInput,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, VisionEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("vision encode");

        let encoder_config = &self.config.encoder_config;
        let patch_size = encoder_config.patch_size;
        let rows = (image.height / patch_size) * (image.width / patch_size);
        let patch_dim = encoder_config.in_channels * encoder_config.temporal_patch_size * patch_size * patch_size;

        let pixels = command_buffer.allocate_constant_from_slice(&image.pixels).map_err(VisionEncodeError::Backend)?;
        let mut patches = command_buffer
            .allocate_scratch_for_shape(&[rows, patch_dim], self.data_type)
            .map_err(VisionEncodeError::Backend)?;
        self.patchify.encode(
            &pixels,
            &mut patches,
            &self.image_mean,
            &self.image_std,
            image.width,
            patch_size,
            encoder_config.spatial_merge_size,
            encoder_config.temporal_patch_size,
            rows * patch_dim,
            command_buffer,
        );
        let stream = self.patch_emb.encode(patches, rows, command_buffer)?;
        Ok(stream)
    }
}
